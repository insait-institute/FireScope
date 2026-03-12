# ---------------------------
# Qwen2.5-VL-7B → Transformer Raster Decoder (drop-in)
# ---------------------------
from typing import List, Union, Optional, Dict, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from transformers import Qwen2_5_VLProcessor, AutoModelForImageTextToText
from PIL import Image as PILImage

# ---------------------------
# Small utilities
# ---------------------------

def _to_pil_list(x: torch.Tensor):
    """
    Convert a (B,C,H,W) float tensor in [0,1] or [-1,1] to a list of PIL Images.
    Qwen's processor prefers PIL or numpy uint8.
    """
    from PIL import Image
    x = x.detach().cpu()
    if x.min() < 0.0:  # support [-1, 1]
        x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (x * 255.0).clamp(0, 255).to(torch.uint8)
    imgs = []
    for i in range(x.size(0)):
        arr = x[i].permute(1, 2, 0).numpy()  # HWC
        imgs.append(Image.fromarray(arr))
    return imgs


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.drop(self.fc1(x)))))


class DecoderBlock(nn.Module):
    """
    Transformer decoder block:
      LN → cross-attn (queries attend to memory) → residual
      LN → MLP → residual
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, q, mem):
        # q: (B, Nq, D)  mem: (B, Nm, D)
        x = q
        qn = self.ln1(q)
        attn_out, _ = self.cross(qn, mem, mem, need_weights=False)
        x = x + attn_out
        y = x + self.mlp(self.ln2(x))
        return y


# ---------------------------
# Main module
# ---------------------------

class QwenVLTransformerRaster(nn.Module):
    """
    Uses Qwen2.5-VL-7B as frozen multimodal encoder (image+text → sequence embeddings),
    then a transformer decoder + conv upsampler to produce (B, out_ch, 341, 341).

    Call: y = model(x, text)
      x: torch.Tensor (B,C,H,W), float in [0,1] or [-1,1]
      text: str or List[str]
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        out_ch: int = 1,
        d_model: int = 512,      # decoder width (matches earlier heads; can increase to 768/1024 if you have VRAM)
        depth: int = 8,          # decoder depth
        heads: int = 8,          # decoder heads
        mlp_ratio: float = 4.0,
        grid_hw: Tuple[int, int] = (43, 43),  # learnable query grid; 43→ upsample to 342, crop to 341
        dtype: torch.dtype = torch.bfloat16,
        freeze_qwen: bool = True
    ):
        super().__init__()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)
        # safety: ensure pad token
        tok = self.processor.tokenizer
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        self.qwen = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.qwen_dtype = dtype

        if freeze_qwen:
            for p in self.qwen.parameters():
                p.requires_grad = False
            self.qwen.eval()

        # Project Qwen hidden size → decoder width
        qwen_hidden = int(self.qwen.config.hidden_size)
        self.mem_proj = nn.Linear(qwen_hidden, d_model)

        # Learnable query grid (Hq×Wq tokens)
        Hq, Wq = grid_hw
        Nq = Hq * Wq
        self.query_tokens = nn.Parameter(torch.randn(1, Nq, d_model) * 0.02)

        # Transformer decoder
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads=heads, mlp_ratio=mlp_ratio, drop=0.0)
            for _ in range(depth)
        ])
        self.dec_ln = nn.LayerNorm(d_model)

        # 2D reshape + upsample conv head to 342×342 → crop to 341×341
        self.Hq, self.Wq = Hq, Wq
        self.ups1 = nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1)  # 43→86
        self.ups2 = nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1)  # 86→172
        self.ups3 = nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1)  # 172→344
        self.conv_refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(d_model // 8, d_model // 8, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(d_model // 8, out_ch, kernel_size=3, padding=1)

        # target before crop
        self._target_hw = (342, 342)

    @torch.no_grad()
    def _encode_qwen(self, images, texts, device):
        """
        images: List[PIL.Image] or List[str paths] length B
        texts:  List[str] length B
        returns last hidden states (B, T, Dq)
        """
        # 1) Build per-sample chat strings that include an <image> placeholder
        chat_texts = []
        for t in texts:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": t}
                ]
            }]
            chat = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,   # ok either way; keeps format consistent
                tokenize=False
            )
            chat_texts.append(chat)

        # 2) Tokenize with images + chat_texts
        enc = self.processor(
            text=chat_texts,     # IMPORTANT: chat-templated strings that contain image tokens
            images=images,       # same length as chat_texts
            return_tensors="pt",
            padding=True
        )

        # 3) Move to device/dtypes Qwen expects
        for k in ("input_ids", "attention_mask"):
            enc[k] = enc[k].to(device)
        enc["pixel_values"] = enc["pixel_values"].to(device, dtype=self.qwen_dtype)
        if "image_grid_thw" in enc:
            enc["image_grid_thw"] = enc["image_grid_thw"].to(device)

        # 4) Forward through Qwen with hidden states
        out = self.qwen(
            **enc,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        # out.hidden_states is a tuple: (hidden_0, hidden_1, ..., hidden_last)
        if out.hidden_states is None:
            raise RuntimeError("Qwen output_hidden_states=None — ensure output_hidden_states=True or set self.qwen.config.output_hidden_states=True.")
        mem = out.hidden_states[-1]   # (B, T, Dq)
        return mem



    def forward(self, image_bchw: Union[torch.Tensor, List[Union[PILImage.Image, str]]],
                text: Union[str, List[str]]):

        device = next(self.parameters()).device  # more robust than image_bchw.device

        # ---- Prepare images (tensor, list of PIL, or list of paths) ----
        if isinstance(image_bchw, torch.Tensor):
            B = image_bchw.size(0)
            images = _to_pil_list(image_bchw)  # convert tensor -> list[PIL.Image]
        elif isinstance(image_bchw, list):
            # list of PIL.Image
            if all(isinstance(im, PILImage.Image) for im in image_bchw):
                images = image_bchw
                B = len(images)
            # or list of file paths (processor can open them)
            elif all(isinstance(im, str) for im in image_bchw):
                images = image_bchw
                B = len(images)
            else:
                raise TypeError("image_bchw list must contain only PIL.Image objects or file path strings.")
        else:
            raise TypeError("image_bchw must be a torch.Tensor (B,C,H,W) or a list of PIL images / file paths.")

        # ---- Prepare texts (broadcast/validate to batch size) ----
        if isinstance(text, str):
            texts = [text] * B
        else:
            assert len(text) == B, f"len(text)={len(text)} must match batch size B={B}"
            texts = text

        # ---- Qwen encoder forward → multimodal sequence memory ----
        with (torch.no_grad() if all(not p.requires_grad for p in self.qwen.parameters())
            else torch.enable_grad()):
            mem = self._encode_qwen(images, texts, device)  # (B, T, Dq)
        # 3) Project memory to decoder width
        mem = mem.to(self.mem_proj.weight.dtype) 
        mem = self.mem_proj(mem)  # (B, T, d_model)

        # 4) Transformer decoder over learnable query grid
        Nq = self.Hq * self.Wq
        q = self.query_tokens.expand(B, Nq, -1).to(mem.dtype) 
        q = self.query_tokens.expand(B, Nq, -1)  # (B, Nq, d_model)
        for blk in self.blocks:
            q = blk(q, mem)
        q = self.dec_ln(q)  # (B, Nq, d_model)

        # 5) Reshape to 2D and upsample to 342×342, then crop to 341×341
        fmap = q.transpose(1, 2).contiguous().view(B, -1, self.Hq, self.Wq)  # (B,d_model,Hq,Wq)
        fmap = fmap.to(self.head.weight.dtype) 
        fmap = self.ups1(fmap)   # 86×86
        fmap = F.gelu(fmap)
        fmap = self.ups2(fmap)   # 172×172
        fmap = F.gelu(fmap)
        fmap = self.ups3(fmap)   # 344×344
        fmap = self.conv_refine(fmap)

        fmap = F.interpolate(fmap, size=self._target_hw, mode='bilinear', align_corners=False)  # 342×342
        y = self.head(fmap)      # (B, out_ch, 342, 342)
        return y[:, :, :-1, :-1] # (B, out_ch, 341, 341)

    # Convenience helpers
    def freeze_qwen(self):
        for p in self.qwen.parameters():
            p.requires_grad = False
        self.qwen.eval()

    def unfreeze_qwen_last_k_decoder_layers(self, k: int = 2):
        """If you decide to fine-tune a bit of Qwen, this unfreezes last k LM blocks."""
        try:
            blocks = self.qwen.model.layers  # common naming
        except AttributeError:
            blocks = self.qwen.transformer.h  # fallback layout
        for p in self.qwen.parameters():
            p.requires_grad = False
        for blk in blocks[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        self.qwen.train()

    def state_dict_unfrozen(self) -> Dict[str, torch.Tensor]:
        """Return only trainable params (keeps checkpoints small if you freeze Qwen)."""
        return {k: v for k, v in self.state_dict().items() if v.requires_grad}
