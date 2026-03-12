import torch as _torch
from config import DATA_DIR
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import glob, random
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import re, json, os

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

_COORD_RE = re.compile(r"lon(?P<lon>-?[0-9]+(?:\.[0-9]+)?)_lat(?P<lat>-?[0-9]+(?:\.[0-9]+)?)")
def parse_lat_lon_from_name_v2(filename: str):
    """
    Extract the last two floating-point numbers from the given filename.
    Returns (lat, lon) as floats.
    """
    # Find all float-like numbers in the string
    numbers = re.findall(r'-?\d+\.\d+', filename)
    
    if len(numbers) < 2:
        raise ValueError("Not enough numeric values found in filename.")
    
    # Last two numbers are latitude and longitude
    lat = float(numbers[-2])
    lon = float(numbers[-1])
    return f"{lat}_{lon}"

def parse_lat_lon_from_name(path: str) -> Optional[str]:
    if 'images_whole_europe' in path:
        return parse_lat_lon_from_name_v2(path)
    m = _COORD_RE.search(Path(path).name)
    if not m:
        return None
    lon = float(m.group('lon'))
    lat = float(m.group('lat'))
    return f"{lat}_{lon}"
from typing import Literal, Union
from PIL import Image as PILImage

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

import pandas

class CustomDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 rast_dir: Optional[str] = None,
                 crop_sat: int = 1023,
                 crop_rast: int = 341,
                 scale: Optional[int] = None,
                 climate_dir: Optional[str] = None,
                 with_cheating: bool = False,
                 # NEW: how to return climate + image
                 climate_return: Literal["vector", "raw"] = "vector",
                 image_return: Literal["tensor", "pil"] = "tensor",
                vlm_pred_json: Optional[str] = None,
                counterfactual: bool = False,
                paraphrase: bool=False,
                targets_index: Optional[str] = None,
                train_img_dir: Optional[str] = '/work/wildfirerisk/small_dataset/satellite_images/train', # used to normalize the climate data
                ):

        if with_cheating and not rast_dir and not vlm_pred_json:
            raise Exception("Need to have ground truth rasters when cheating is enabled without oracle!")
        self.img_paths = sorted(glob.glob(str(Path(img_dir) / '*.png')))
        if not self.img_paths:
            raise FileNotFoundError(f"No PNGs in {img_dir}")
        if targets_index:
            tar_fnames = set()
            csvFile = pandas.read_csv(targets_index)
            for idx, tl in csvFile.iterrows():
                l = tl.to_dict()
                minx, miny, maxx, maxy = l['minx'], l['miny'], l['maxx'], l['maxy']
                lat, lon = l['lat_centroid'], l['lon_centroid']
                tar_fnames.add(f'tile_{minx}_{miny}_{maxx}_{maxy}_{lat}_{lon}.png')
            self.img_paths = [x for x in self.img_paths if Path(x).name in tar_fnames]
        self.with_raster = rast_dir is not None
        if self.with_raster:
            self.rast_dir = Path(rast_dir)
        self.counterfactual=counterfactual
        self.paraphrase=paraphrase
        self.vlm_preds = None
        if vlm_pred_json is not None:
            with open(vlm_pred_json, "r") as f:
                preds = json.load(f)  # expected: { "lat_lon": <digit or scalar> }
            if not isinstance(preds, dict):
                raise ValueError(f"vlm_pred_json must contain a JSON dict, got {type(preds)}")
            self.vlm_preds = preds  # keep raw; we’ll interpret per-sample

        self.with_climate = climate_dir is not None
        if self.with_climate:
            # use all images in this folder as reference keys for norm
            train_imgs = [parse_lat_lon_from_name(fname) for fname in os.listdir(train_img_dir)]
            self.climate_raw, self.climate_vectors = load_climate_vectors_and_raw(
                climate_dir, ref_for_norm=train_imgs
            )

        self.with_cheating = with_cheating
        self.crop_sat = crop_sat
        self.crop_rast = crop_rast
        self.scale = scale or (crop_sat // crop_rast)
        assert self.scale * crop_rast == crop_sat, "crop sizes must be integer-multiple aligned"

        self.climate_return = climate_return
        self.image_return = image_return

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base = Path(img_path).stem
        if self.with_raster:
            if 'split_satellite_images' not in img_path:
                rast_path = self.rast_dir / f"{base}.npy"
            else:
                rast_path = self.rast_dir / f"{base[:-6]}.npy"
            if not rast_path.is_file():
                raise FileNotFoundError(f"Missing raster for {img_path}: {rast_path}")

        # load full-res
        img_pil = PILImage.open(img_path).convert('RGB')
        img_np = np.asarray(img_pil, dtype=np.float32) / 255.0  # [H,W,3]
        if self.with_raster:
            rast_np = np.load(rast_path).astype(np.float32)
            if rast_np.ndim == 3:
                rast_np = rast_np[0]
        Hr, Wr, _ = img_np.shape
        Hr = Hr // 3
        Wr = Wr // 3
        top_r = random.randint(0, Hr - self.crop_rast)
        left_r = random.randint(0, Wr - self.crop_rast)
        top_s, left_s = top_r * self.scale, left_r * self.scale

        # crop
        if 'split_satellite_images' in img_path:
            row = int(base[-4])
            col = int(base[-1])
            if self.with_raster:
                rast_crop = rast_np[row*self.crop_rast:(row+1)*self.crop_rast, col*self.crop_rast:(col+1)*self.crop_rast]
            if self.image_return == "pil":
                img_crop = img_pil
            else:
                img_crop = img_np
        else:
            if self.image_return == "pil":
                img_crop = img_pil.crop((left_s, top_s, left_s + self.crop_sat, top_s + self.crop_sat))
            else:
                img_crop = img_np[top_s:top_s+self.crop_sat, left_s:left_s+self.crop_sat, :]
            if self.with_raster:
                rast_crop = rast_np[top_r:top_r+self.crop_rast, left_r:left_r+self.crop_rast]
        if self.with_raster:
            rast_clip = np.clip(rast_crop, 0.0, 1.0)
            rast_norm = (rast_clip - 0.5) / 0.5

        # climate
        if self.with_climate:
            key = parse_lat_lon_from_name(img_path)
            if key is None:
                raise KeyError(f"Could not parse climate key from {img_path}")
            if self.climate_return == "raw":
                cond = self.climate_raw[key]  # dict
            else:
                cond = torch.from_numpy(self.climate_vectors[key]).contiguous().float()  # tensor
        else:
            cond = torch.tensor([-1.0], dtype=torch.float32)

        # tensors
        if self.image_return == "tensor":
            img_t = torch.from_numpy(img_crop).permute(2,0,1).contiguous().float()  # [3,H,W] in [0,1]
        else:
            img_t = img_crop  # PIL.Image.Image

        if self.with_raster:
            rast_t = torch.from_numpy(rast_norm)[None, ...].contiguous().float()
        else:
            rast_t = torch.zeros((Hr, Wr))
        # --- cheating / conditioning source ---
        if self.with_cheating:
            if self.vlm_preds is not None:
                key = parse_lat_lon_from_name(img_path)
                if 'split_satellite_images' in img_path:
                    row_col = img_path.split(".")[-2].split("_")[-2:]
                    row_col = "_".join(row_col)
                    key += '_' + row_col
                if key is None or key not in self.vlm_preds:
                    raise KeyError(f"VLM pred missing for key '{key}' from {img_path}")
                if self.counterfactual:
                    val = self.vlm_preds[key]['counterfactual_answer']
                elif self.paraphrase:
                    val = self.vlm_preds[key]['paraphrased_answer']
                else:
                    val = self.vlm_preds[key]['answer']
                
                # Accept int/str digit (0..9) or already-normalized scalar
                if isinstance(val, (int, np.integer)) or (isinstance(val, str) and val.isdigit()):
                    cheat_scalar = digit_to_cond_scalar(int(val))
                else:
                    cheat_scalar = float(val)  # assume already in [-1,1]
                # clamp just in case
                cheat_scalar = max(-1.0, min(1.0, cheat_scalar))
                cheat = torch.tensor([cheat_scalar], dtype=torch.float32)
            else:
                # default: use GT mean from raster
                cheat = rast_t.mean((-1, -2))
        else:
            cheat = torch.tensor([-1.0], dtype=torch.float32)


        return img_t, cond, rast_t, cheat, base

def collate_vlm(batch):
    """
    Allows cond to be either a tensor (vectors) or a dict (raw).
    Also supports images as tensors or PIL.
    """
    imgs, conds, rasts, cheats, bases = zip(*batch)

    # images can be list(PIL) or stacked tensor
    if isinstance(imgs[0], PILImage.Image):
        imgs = list(imgs)
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
