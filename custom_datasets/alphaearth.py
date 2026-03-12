from config import DATA_DIR
from pathlib import Path
import os, re, glob, json
from typing import Optional, List, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


def digit_to_cond_scalar(d: int) -> float:
    """
    Map 0..9 digit to the conditioning scalar in [-1, 1].
    This matches the midpoint-per-bin mapping used earlier.
    """
    return ((int(d) + 0.5) / 10.0 - 0.5) / 0.5  # == (d+0.5)/5 - 1

def dict_to_vector(data):
    """
    Recursively flattens a nested dict into a deterministic numeric vector.
    Keys are sorted alphabetically at each level to ensure consistent order.
    """
    def flatten(d):
        items = []
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                items.extend(flatten(value))
            else:
                items.append(value)
        return items

    return flatten(data)

def load_climate_vectors_and_raw(climate_json: str, ref_for_norm: list):
    with open(climate_json, 'r') as f:
        raw = json.load(f)  # {key: dict}
    # vectorize
    keys = []
    mats, mats_for_norm = [], []
    for k, v in raw.items():
        vec = np.asarray(dict_to_vector(v), dtype=np.float32)
        mats.append(vec); keys.append(k)
        if k in ref_for_norm:
            mats_for_norm.append(vec)
    mats = np.stack(mats, axis=0)
    mats_for_norm = np.stack(mats_for_norm, axis=0)
    assert mats.shape[1] == 60, f"cond_dim mismatch: got {mats.shape[1]}, expected 60"
    mu = mats_for_norm.mean(0, keepdims=True)
    sd = mats_for_norm.std(0, keepdims=True) + 1e-6
    mats = (mats - mu) / sd
    vec_map = {k: mats[i] for i, k in enumerate(keys)}
    return raw, vec_map

_COORD_RE = re.compile(r"lon(?P<lon>-?[0-9]+(?:\.[0-9]+)?)_lat(?P<lat>-?[0-9]+(?:\.[0-9]+)?)")

def parse_lat_lon_from_name(path: str) -> Optional[str]:
    m = _COORD_RE.search(Path(path).name)
    if not m:
        return None
    lon = float(m.group('lon'))
    lat = float(m.group('lat'))
    return f"{lat}_{lon}"

class CustomDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 rast_dir: str,
                 cond_json: str = None,
                 image_return: Literal["tensor", "ndarray"] = "tensor",
                 train_img_dir: Optional[str] = None,
                ):
        self.img_paths = sorted(glob.glob(str(Path(img_dir) / '*.npy')))
        if not self.img_paths:
            raise FileNotFoundError(f"No NPYs in {img_dir}")
        self.rast_dir = Path(rast_dir)
        is_climate = 'climat' in cond_json

        if cond_json:
            with open(cond_json, 'r') as f:
                self.cond_json = json.load(f)
            if not is_climate:
                conded_rsters = [Path(r['raster_path']) for ll, r in self.cond_json.items()]
                self.img_paths = [im for im in self.img_paths if self.rast_dir / f"{Path(im).stem}.npy" in conded_rsters]
            # else:
            #     conded_imgs = [Path(r['img_path']).stem for ll, r in self.cond_json.items()]
            #     self.img_paths = [im for im in self.img_paths if Path(im).stem in conded_imgs]

        else:
            self.cond_json = None
        self.image_return = image_return

        self.with_climate = is_climate
        if self.with_climate:
            # use all images in this folder as reference keys for norm
            train_imgs = [parse_lat_lon_from_name(fname) for fname in os.listdir(train_img_dir)]
            self.climate_raw, self.climate_vectors = load_climate_vectors_and_raw(
                cond_json, ref_for_norm=train_imgs
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base = Path(img_path).stem
        rast_path = self.rast_dir / f"{base}.npy"
        if not rast_path.is_file():
            raise FileNotFoundError(f"Missing raster for {img_path}: {rast_path}")

        # load full-res
        img_np = np.load(img_path).astype(np.float32)
        rast_np = np.load(rast_path).astype(np.float32)
        if rast_np.ndim == 3:
            rast_np = rast_np[0]

        rast_clip = np.clip(rast_np, 0.0, 1.0)
        rast_norm = (rast_clip - 0.5) / 0.5

        # tensors
        if self.image_return == "tensor":
            img_t = torch.from_numpy(img_np).contiguous().float()  # [64,H,W] normalised
        else:
            img_t = img_np  # np.ndarray

        rast_t = torch.from_numpy(rast_norm)[None, ...].contiguous().float()
        # --- cheating / conditioning source ---
        cheat = torch.tensor([-1.0], dtype=torch.float32)

        # cond
        if self.cond_json:
            if self.with_climate:
                key = parse_lat_lon_from_name(img_path)
                if key is None:
                    raise KeyError(f"Could not parse climate key from {img_path}")
                cond = torch.from_numpy(self.climate_vectors[key]).contiguous().float()  # tensor
            else:
                cond = torch.tensor(self.cond_json[parse_lat_lon_from_name(base)]['answer']/10.0, dtype=torch.float32)
        else:
            cond = cheat

        return img_t, cond, rast_t, cheat, base

def collate_vlm(batch):
    """
    Allows cond to be either a tensor (vectors) or a dict (raw).
    Also supports images as tensors or PIL.
    """
    imgs, conds, rasts, cheats, bases = zip(*batch)

    # images can be list(PIL) or stacked tensor
    if isinstance(imgs[0], np.ndarray):
        imgs = np.stack(imgs, 0)
    else:
        imgs = torch.stack(imgs, dim=0)

    rasts = torch.stack(rasts, dim=0)
    cheats = torch.stack([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in cheats], dim=0)
    bases = list(bases)

    # conds: either stack tensors or keep list of dicts
    if conds[0] is None:
        conds_out = None
    elif isinstance(conds[0], torch.Tensor):
        conds_out = torch.stack(conds, dim=0)
    else:
        conds_out = list(conds)  # list of dicts

    return imgs, conds_out, rasts, cheats, bases
