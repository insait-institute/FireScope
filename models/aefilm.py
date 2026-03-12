
from typing import Optional, Dict
import torch
from torch import nn


class FiLMGroupNorm(nn.Module):
    """LayerNorm where affine params are produced per-sample via FiLM (gamma,beta)."""
    def __init__(self, groups, shape):
        super().__init__()
        self.norm = nn.GroupNorm(groups, shape)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        y = self.norm(x)
        return y * gamma + beta


class AEFiLM(nn.Module):
    def __init__(self,
                 in_ch: int = 64,
                 out_ch: int = 1,
                 base: int = 64,
                 cond_dim: int = 60):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2, bias=False),
            nn.LayerNorm(cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, 2 * base, bias=True),
        )

        self.conv = nn.Conv2d(in_ch, base, 3, padding=1, bias=False)
        self.norm = FiLMGroupNorm(8, base)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(base, out_ch, 3, padding=1, bias=True)
        )


    def forward(self,
                image_bchw: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                return_latent: bool = False,
                no_decoder: bool = False):

        z = self.conv(image_bchw)
        B, D, Ht, Wt = z.shape

        # 2) Conditioning
        if cond is None:
            cond = torch.zeros(B, self.film_gen[0].in_features, device=z.device, dtype=z.dtype)
        elif len(cond.shape)==1:
            cond = cond.unsqueeze(1)
        film_vec = self.film_gen(cond)
        film = film_vec.view(B, 2, D)
        gamma = film[:, 0, :, None, None]
        beta  = film[:, 1, :, None, None]

        z = self.norm(z, gamma, beta)

        y = self.head(z)

        return y

    def state_dict_unfrozen(self) -> Dict[str, torch.Tensor]:
        return self.state_dict()