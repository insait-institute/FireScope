# ---------------------------
# SegFormer (MIT-B5) + OOM-friendly Decoder + FiLM
# Output matches your UNet: 341 x 341
# ---------------------------
from typing import Optional, Dict, List
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import SegformerModel

# ---- FiLM ----
def film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    beta = beta.view(x.size(0), x.size(1), 1, 1)
    gamma = gamma.view(x.size(0), x.size(1), 1, 1)
    return gamma * x + beta

# ---- 2D sine-cosine PE ----
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("PositionalEncoding2D channels must be divisible by 4.")
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        pe = self._build_pe(h, w, device)
        if pe.size(1) != c:
            if pe.size(1) < c:
                pe = F.pad(pe, (0,0,0,0,0, c - pe.size(1)))
            else:
                pe = pe[:, :c]
        return x + pe

    def _build_pe(self, h: int, w: int, device) -> torch.Tensor:
        c = self.channels
        cq = c // 4
        ch = c // 2
        y = torch.linspace(0, 1, steps=h, device=device).unsqueeze(1).repeat(1, w)
        x = torch.linspace(0, 1, steps=w, device=device).unsqueeze(0).repeat(h, 1)
        divy = torch.exp(torch.arange(0, cq, device=device) * (-math.log(10000.0) / cq))
        divx = torch.exp(torch.arange(0, cq, device=device) * (-math.log(10000.0) / cq))
        pey = torch.zeros(h, w, ch, device=device)
        pex = torch.zeros(h, w, ch, device=device)
        pey[..., 0::2] = torch.sin(y.unsqueeze(-1) * divy)
        pey[..., 1::2] = torch.cos(y.unsqueeze(-1) * divy)
        pex[..., 0::2] = torch.sin(x.unsqueeze(-1) * divx)
        pex[..., 1::2] = torch.cos(x.unsqueeze(-1) * divx)
        return torch.cat([pey, pex], dim=-1).permute(2,0,1).unsqueeze(0)  # (1,C,H,W)

# ---- 1x1 + GN ----
class ConvProject(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = nn.GroupNorm(8, out_ch)
    def forward(self, x):
        return self.norm(self.proj(x))

# ---- Transformer decoder stage ----
class XAttnDecoderStage(nn.Module):
    def __init__(self, enc_ch: int, d_model: int = 384, nhead: int = 6, num_layers: int = 3, use_checkpoint: bool = False):
        super().__init__()
        self.mem_proj = ConvProject(enc_ch, d_model)
        self.tgt_norm = nn.GroupNorm(8, d_model)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=False)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pos = PositionalEncoding2D(d_model)
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model + d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, d_model),
        )
        self.use_checkpoint = use_checkpoint

    def _run_decoder(self, tgt_seq, mem_seq):
        return self.decoder(tgt=tgt_seq, memory=mem_seq)

    def forward(self, enc_feat: torch.Tensor, tgt_grid: torch.Tensor) -> torch.Tensor:
        B, _, H, W = enc_feat.shape
        mem = self.mem_proj(enc_feat)           # (B,d,H,W)
        tgt = self.tgt_norm(self.pos(tgt_grid)) # (B,d,H,W)
        mem_seq = mem.flatten(2).permute(2,0,1) # (HW,B,d)
        tgt_seq = tgt.flatten(2).permute(2,0,1) # (HW,B,d)
        if self.use_checkpoint:
            out_seq = torch.utils.checkpoint.checkpoint(self._run_decoder, tgt_seq, mem_seq, use_reentrant=False)
        else:
            out_seq = self._run_decoder(tgt_seq, mem_seq)
        out = out_seq.permute(1,2,0).reshape(B, mem.size(1), H, W)
        return self.fuse(torch.cat([out, mem], dim=1))

# ---- High-res conv-only fuse (1/4) ----
class ConvFuseStage(nn.Module):
    def __init__(self, enc_ch: int, d_model: int):
        super().__init__()
        self.enc_proj = nn.Conv2d(enc_ch, d_model, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, d_model),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, d_model),
        )
    def forward(self, enc_feat: torch.Tensor, up_feat: torch.Tensor) -> torch.Tensor:
        mem = self.enc_proj(enc_feat)
        return self.fuse(torch.cat([up_feat, mem], dim=1))

# ---- Full model ----
class FilmSegformer(nn.Module):
    """
    SegFormer MIT-B5 encoder + OOM-friendly decoder + FiLM
    Output: 341x341 (matches your UNet)
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        cond_dim: int = 256,
        d_model: int = 384,
        decoder_layers_per_stage: int = 3,
        nhead: int = 6,
        use_pretrained: bool = True,
        use_checkpoint_dec: bool = False,
    ):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained("nvidia/mit-b5") if use_pretrained \
            else SegformerModel.from_config(SegformerModel.from_pretrained("nvidia/mit-b5").config)

        self.enc_channels = [64, 128, 320, 512]  # SegFormer B5
        self.d_model = d_model
        self.out_ch = out_ch

        # Decoder stages: 1/32, 1/16, 1/8 use transformer; 1/4 conv-only
        self.dec4 = XAttnDecoderStage(self.enc_channels[3], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec3 = XAttnDecoderStage(self.enc_channels[2], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec2 = XAttnDecoderStage(self.enc_channels[1], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec1 = ConvFuseStage(self.enc_channels[0], d_model)

        # Head: project at 1/4, then upsample directly to 1/3 of padded size and predict
        self.head_pre = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, d_model // 2),
        )
        self.head_out = nn.Conv2d(d_model // 2, out_ch, 3, padding=1)

        # FiLM: 4 enc sites (raw chans) + 4 dec sites (d_model)
        film_dims_enc = self.enc_channels
        film_dims_dec = [d_model, d_model, d_model, d_model]
        self._film_dims = film_dims_enc + film_dims_dec
        self._film_slices = [slice(sum(self._film_dims[:i]), sum(self._film_dims[:i+1])) for i in range(len(self._film_dims))]
        self._film_total = sum(self._film_dims)
        self.film_generator = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2, bias=False),
            nn.LayerNorm(cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, self._film_total * 2, bias=True),
        )

        # no_decoder latent (project 1/8 to d_model)
        self.no_dec_proj = ConvProject(self.enc_channels[1], d_model)
        
        # Seed for coarsest grid (1/32)
        self.seed_token = nn.Parameter(torch.randn(1, d_model, 1, 1))

    # ---- helpers ----
    def _apply_film_idx(self, x: torch.Tensor, film_vec: torch.Tensor, idx: int) -> torch.Tensor:
        s = self._film_slices[idx]
        gamma = film_vec[:, s, 0]; beta = film_vec[:, s, 1]
        return film(x, gamma, beta)

    # ---- forward ----
    def forward(self, image_bchw: torch.Tensor, cond: Optional[torch.Tensor] = None,
                return_latent: bool = False, no_decoder: bool = False):
        # Reflect-pad to 1026×1026 so UNet/SegFormer paths line up neatly
        x1026 = F.pad(image_bchw, (1, 2, 1, 2), mode="reflect")

        # FiLM params
        if cond is None:
            cond = torch.zeros(x1026.size(0), self.film_generator[0].in_features,
                               device=x1026.device, dtype=x1026.dtype)
        film_vec = self.film_generator(cond).view(x1026.size(0), self._film_total, 2)

        # Encoder (4 scales)
        enc_out = self.encoder(pixel_values=x1026, output_hidden_states=True, return_dict=True)
        hs = enc_out.hidden_states[-4:]

        enc_feats: List[torch.Tensor] = []
        for i, h in enumerate(hs):
            if h.dim() == 4:
                feat = h  # (B,C,H,W)
            else:
                B, HW, C = h.shape
                H_in, W_in = x1026.shape[-2:]
                scale = 2 ** (i + 2)  # 1/4,1/8,1/16,1/32
                H, W = math.ceil(H_in / scale), math.ceil(W_in / scale)
                feat = h.transpose(1, 2).reshape(B, C, H, W)
            enc_feats.append(feat)

        # FiLM on raw encoder features (sites 0..3)
        for i in range(4):
            enc_feats[i] = self._apply_film_idx(enc_feats[i], film_vec, i)

        # Early exit: ~1/8 projected latent
        if no_decoder:
            return self.no_dec_proj(enc_feats[1])

        # Decoder pyramid
        B, _, H4, W4 = enc_feats[3].shape
        y4_seed = self.seed_token.expand(B, self.d_model, H4, W4)
        y4 = self.dec4(enc_feats[3], y4_seed)
        y4 = self._apply_film_idx(y4, film_vec, 4)

        _, _, H3, W3 = enc_feats[2].shape
        y3_seed = F.interpolate(y4, size=(H3, W3), mode="bilinear", align_corners=False)
        y3 = self.dec3(enc_feats[2], y3_seed)
        y3 = self._apply_film_idx(y3, film_vec, 5)

        _, _, H2, W2 = enc_feats[1].shape
        y2_seed = F.interpolate(y3, size=(H2, W2), mode="bilinear", align_corners=False)
        y2 = self.dec2(enc_feats[1], y2_seed)
        y2 = self._apply_film_idx(y2, film_vec, 6)

        _, _, H1, W1 = enc_feats[0].shape
        y1_seed = F.interpolate(y2, size=(H1, W1), mode="bilinear", align_corners=False)
        y1 = self.dec1(enc_feats[0], y1_seed)
        y1 = self._apply_film_idx(y1, film_vec, 7)  # still at ~1/4

        # ----- NEW HEAD: 1/4 -> 1/3 (342x342 for 1026x1026), then crop to 341x341 -----
        target_h = x1026.shape[-2] // 3  # 1026//3 = 342
        target_w = x1026.shape[-1] // 3  # 1026//3 = 342
        y = self.head_pre(y1)
        y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
        y = self.head_out(y)
        out = y[:, :, :-1, :-1]  # 341 x 341 (matches your UNet)
        return out

    def state_dict_unfrozen(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()

class SegformerNoFilm(nn.Module):
    """
    SegFormer MIT-B5 encoder + OOM-friendly decoder (NO FiLM)
    Output: 341 x 341 (matches your UNet and FilmSegformer sizing)
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        d_model: int = 384,
        decoder_layers_per_stage: int = 3,
        nhead: int = 6,
        use_pretrained: bool = True,
        use_checkpoint_dec: bool = False,
    ):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained("nvidia/mit-b5") if use_pretrained \
            else SegformerModel.from_config(SegformerModel.from_pretrained("nvidia/mit-b5").config)

        self.enc_channels = [64, 128, 320, 512]  # SegFormer B5
        self.d_model = d_model
        self.out_ch = out_ch

        # Decoder stages: 1/32, 1/16, 1/8 use transformer; 1/4 conv-only
        self.dec4 = XAttnDecoderStage(self.enc_channels[3], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec3 = XAttnDecoderStage(self.enc_channels[2], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec2 = XAttnDecoderStage(self.enc_channels[1], d_model, nhead, decoder_layers_per_stage, use_checkpoint_dec)
        self.dec1 = ConvFuseStage(self.enc_channels[0], d_model)

        # Head: project at 1/4, then upsample directly to 1/3 of padded size and predict
        self.head_pre = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, d_model // 2),
        )
        self.head_out = nn.Conv2d(d_model // 2, out_ch, 3, padding=1)

        # no_decoder latent (project 1/8 to d_model)
        self.no_dec_proj = ConvProject(self.enc_channels[1], d_model)

        # Seed for coarsest grid (1/32)
        self.seed_token = nn.Parameter(torch.randn(1, d_model, 1, 1))

    def forward(self, image_bchw: torch.Tensor, return_latent: bool = False, no_decoder: bool = False):
        # Reflect-pad to 1026×1026 so paths line up neatly
        x1026 = F.pad(image_bchw, (1, 2, 1, 2), mode="reflect")

        # Encoder (4 scales)
        enc_out = self.encoder(pixel_values=x1026, output_hidden_states=True, return_dict=True)
        hs = enc_out.hidden_states[-4:]

        enc_feats = []
        for i, h in enumerate(hs):
            if h.dim() == 4:
                feat = h  # (B,C,H,W)
            else:
                B, HW, C = h.shape
                H_in, W_in = x1026.shape[-2:]
                scale = 2 ** (i + 2)  # 1/4, 1/8, 1/16, 1/32
                H, W = math.ceil(H_in / scale), math.ceil(W_in / scale)
                feat = h.transpose(1, 2).reshape(B, C, H, W)
            enc_feats.append(feat)

        # Early exit: ~1/8 projected latent
        if no_decoder:
            return self.no_dec_proj(enc_feats[1])

        # Decoder pyramid (no FiLM)
        B, _, H4, W4 = enc_feats[3].shape
        y4_seed = self.seed_token.expand(B, self.d_model, H4, W4)
        y4 = self.dec4(enc_feats[3], y4_seed)

        _, _, H3, W3 = enc_feats[2].shape
        y3_seed = F.interpolate(y4, size=(H3, W3), mode="bilinear", align_corners=False)
        y3 = self.dec3(enc_feats[2], y3_seed)

        _, _, H2, W2 = enc_feats[1].shape
        y2_seed = F.interpolate(y3, size=(H2, W2), mode="bilinear", align_corners=False)
        y2 = self.dec2(enc_feats[1], y2_seed)

        _, _, H1, W1 = enc_feats[0].shape
        y1_seed = F.interpolate(y2, size=(H1, W1), mode="bilinear", align_corners=False)
        y1 = self.dec1(enc_feats[0], y1_seed)  # still at ~1/4

        # Head: 1/4 -> 1/3 (342x342 for 1026x1026), then crop to 341x341
        target_h = x1026.shape[-2] // 3  # 1026//3 = 342
        target_w = x1026.shape[-1] // 3  # 1026//3 = 342
        y = self.head_pre(y1)
        y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
        y = self.head_out(y)
        out = y[:, :, :-1, :-1]  # 341 x 341
        return out

    def state_dict_unfrozen(self):
        return self.state_dict()
