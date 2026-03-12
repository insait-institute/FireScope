# ---------------------------
# UNet Model
# ---------------------------

from typing import Optional, Dict

import torch
from torch import nn, einsum
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch, up=False):
        super().__init__()
        in_ch = 2*ch if up else ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
        self.skip = nn.Conv2d(in_ch, ch, 1) if up else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))

class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        # Input expected 1026x1026 (we'll pad 1023→1026 before calling)
        self.down1 = nn.Sequential(  # 1026 -> 342
            nn.Conv2d(in_ch, base, 7, stride=3, padding=3),
            nn.GELU(),
            ResBlock(base),
        )
        self.down2 = nn.Sequential(  # 342 -> 171
            nn.Conv2d(base, base*2, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(base*2),
        )
        self.down3 = nn.Sequential(  # 171 -> 86
            nn.Conv2d(base*2, base*4, 4, stride=2, padding=2),
            nn.GELU(),
            ResBlock(base*4),
        )
        self.down4 = nn.Sequential(  # 86 -> 43
            nn.Conv2d(base*4, base*8, 4, stride=2, padding=1),
            nn.GELU(),
            ResBlock(base*8),
        )

        self.up1 = nn.Sequential(  # 43 -> 86
            nn.ConvTranspose2d(base*8, base*4, 4, stride=2, padding=1),
            nn.GELU()
        )
        self.upres1 = ResBlock(base*4, up=True)
        self.up2 = nn.Sequential(  # 86 -> 172
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.upres2 = ResBlock(base*2, up=True)
        self.up3 = nn.Sequential(  # 171 -> 342
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.upres3 = ResBlock(base, up=True)

        self.out = nn.Conv2d(base, out_ch, 3, padding=1)  # -> 344x344

    def forward(self, image_bchw: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                return_latent: bool = False,
                no_decoder: bool = False
            ):
        # pad to 1026x1026 to make down/up exact, then crop back at the end
        x1026 = F.pad(image_bchw, (1,2,1,2), mode='reflect')  # (left,right,top,bottom)
        # encoder pass
        x342 = self.down1(x1026)
        x171 = self.down2(x342)
        x86 = self.down3(x171)
        x43 = self.down4(x86)
        # decoder pass
        y86 = self.upres1(torch.cat((x86, self.up1(x43)), 1))
        if no_decoder:
            return y86
        else:
            y171 = self.upres2(torch.cat((x171, self.up2(y86)[:, :, :-1, :-1]), 1))
            y342 = self.upres3(torch.cat((x342, self.up3(y171)), 1))
            y = self.out(y342)
            return y[:, :, :-1, :-1]

    def state_dict_unfrozen(self) -> Dict[str, torch.Tensor]:
        """
        Merge state_dicts of unfrozen modules so you can save just the trainable parts.
        """
        return self.state_dict()

def film(x, gamma, beta):
    beta = beta.view(x.size(0), x.size(1), 1, 1)
    gamma = gamma.view(x.size(0), x.size(1), 1, 1)
    x = gamma * x + beta
    return x

class FilmUnet(Unet):
    def __init__(self, in_ch=3, out_ch=1, base=64, cond_dim=256):
        super().__init__(in_ch=in_ch, out_ch=out_ch, base=base)
        film_channels = [ch * base for ch in [1, 2, 4, 8, 4, 2, 1]]
        self.film_slices = [slice(sum(film_channels[:i]), sum(film_channels[:i+1])) for i in range(len(film_channels))]
        self.film_channels = sum(film_channels)
        self.film_generator = nn.Sequential(
            nn.Linear(cond_dim, cond_dim*2, bias=False),
            nn.LayerNorm(cond_dim*2),
            nn.GELU(),
            nn.Linear(cond_dim*2, self.film_channels * 2, bias=True)
        )

    def forward(self, image_bchw: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                return_latent: bool = False,
                no_decoder: bool = False
            ):
        # pad to 1026x1026 to make down/up exact, then crop back at the end
        x1026 = F.pad(image_bchw, (1,2,1,2), mode='reflect')  # (left,right,top,bottom)
        # generate film params
        film_vector = self.film_generator(cond).view(x1026.size(0), self.film_channels, 2)

        # encoder pass
        x342 = film(self.down1(x1026), film_vector[:, self.film_slices[0], 0], film_vector[:, self.film_slices[0], 1])
        x171 = film(self.down2(x342), film_vector[:, self.film_slices[1], 0], film_vector[:, self.film_slices[1], 1])
        x86 = film(self.down3(x171), film_vector[:, self.film_slices[2], 0], film_vector[:, self.film_slices[2], 1])
        x43 = film(self.down4(x86), film_vector[:, self.film_slices[3], 0], film_vector[:, self.film_slices[3], 1])
        # decoder pass
        y86 = self.upres1(torch.cat((x86, self.up1(x43)), 1))
        if no_decoder:
            return y86
        else:
            y86 = film(y86, film_vector[:, self.film_slices[4], 0], film_vector[:, self.film_slices[4], 1])
            y171 = self.upres2(torch.cat((x171, self.up2(y86)[:, :, :-1, :-1]), 1))
            y171 = film(y171, film_vector[:, self.film_slices[5], 0], film_vector[:, self.film_slices[5], 1])
            y342 = self.upres3(torch.cat((x342, self.up3(y171)), 1))
            y342 = film(y342, film_vector[:, self.film_slices[6], 0], film_vector[:, self.film_slices[6], 1])
            y = self.out(y342)
            return y[:, :, :-1, :-1]