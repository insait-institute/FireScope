import os, re, sys, math, json, time, random, glob
project_dir = f'/home/{os.environ["USER"]}/FireScope'
os.chdir(project_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import torch as _torch
from custom_datasets.alphaearth import CustomDataset
from config import DATA_DIR
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from models.aefilm import AEFiLM

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_distributed():
    # torchrun sets these environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank

def is_main_process():
    return int(os.environ.get("RANK", 0)) == 0


def finite_diff_gradients(x):
    # x: [B,1,H,W]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    # pad to original size
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return dx, dy


class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=1, sigma=1.5):
        super().__init__()
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        gauss = (gauss/gauss.sum()).unsqueeze(0)
        window = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        self.register_buffer('window', window)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1, img2):
        window = self.window.to(dtype=torch.float32, device=img1.device)
        img1 = img1.float(); img2 = img2.float()
        mu1 = F.conv2d(img1, window, padding=window.shape[-1]//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window.shape[-1]//2, groups=1)
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window.shape[-1]//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window.shape[-1]//2, groups=1) - mu2_sq
        sigma12   = F.conv2d(img1*img2, window, padding=window.shape[-1]//2, groups=1) - mu1_mu2
        ssim_map  = ((2*mu1_mu2 + self.C1)*(2*sigma12 + self.C2))/((mu1_sq + mu2_sq + self.C1)*(sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean()


def _to_uint8_inverted_gray(img_01):
    """
    img_01: torch.Tensor/np.ndarray in [0,1], shape [H,W] or [1,H,W] or [B,1,H,W].
    Returns a uint8 numpy (H,W) where 0->white, 1->black (as in your prior eval).
    """
    if isinstance(img_01, torch.Tensor):
        x = img_01.detach().cpu().float().clone()
        if x.ndim == 4:   # [B,1,H,W]
            x = x[0, 0]
        elif x.ndim == 3: # [1,H,W]
            x = x[0]
        x = x.clamp(0, 1).numpy()
    else:
        x = np.asarray(img_01, dtype=np.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
    x = 1.0 - x
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x


def _save_side_by_side_png(gt_01, pred_01, out_path):
    left  = _to_uint8_inverted_gray(gt_01)
    right = _to_uint8_inverted_gray(pred_01)
    H, W = left.shape
    canvas = Image.new('L', (W*2, H))
    canvas.paste(Image.fromarray(left), (0, 0))
    canvas.paste(Image.fromarray(right), (W, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------
# EFD model (loaded exactly as instructed)
# ---------------------------

def build_model_and_device(local_rank: int):

    device = _torch.device(f'cuda:{local_rank}' if _torch.cuda.is_available() else 'cpu')

    model = AEFiLM(cond_dim=1)

    model.to(device)
    return model, device



# ---------------------------
# Training & Eval
# ---------------------------

@dataclass
class Cfg:
    train_img_dir: str
    train_rast_dir: str
    val_img_dir: str
    val_rast_dir: str
    climate_json: str
    out_dir: str
    eval_every: int
    epochs: int = 100
    batch_size: int = 8
    accum_steps: int = 1
    num_workers: int = 6
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    log_every: int = 25
    seed: int = 42

    w_recon: float = 1.0
    w_ssim: float = 0.5
    w_grad: float = 0.2
    huber_delta: float = 1.0


def grad_loss(pred, target):
    dx_p, dy_p = finite_diff_gradients(pred)
    dx_t, dy_t = finite_diff_gradients(target)
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def evaluate(model, device, val_loader, ssim, huber, cfg: Cfg) -> Dict[str, float]:
    model.eval()
    totals = {"recon":0.0, "ssim":0.0, "grad":0.0}
    n = 0
    with torch.no_grad():
        for img, climate, rast, cheat, base in val_loader:
            img = img.to(device, non_blocking=True)
            cond = climate.to(device, non_blocking=True)
            rast = rast.to(device, non_blocking=True)
            pred = model(img, cond=cond)
            l_recon = huber(pred, rast)
            pred_01 = (pred * 0.5 + 0.5).clamp(0,1)
            rast_01 = (rast * 0.5 + 0.5).clamp(0,1)
            l_ssim = 1.0 - ssim(pred_01, rast_01)
            l_grad = grad_loss(pred, rast)
            bs = img.size(0)
            totals["recon"] += l_recon.item()*bs
            totals["ssim"]  += l_ssim.item()*bs
            totals["grad"]  += l_grad.item()*bs
            n += bs

    # Distributed average
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([totals["recon"], totals["ssim"], totals["grad"], float(n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        totals["recon"], totals["ssim"], totals["grad"], n = t.tolist()

    return {k: v/max(n,1) for k,v in totals.items()}



def save_val_samples_fixed(model, device, val_dataset: CustomDataset, indices: List[int], out_dir: str, epoch: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(12345)  # keep for any tie-breaking
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices, start=1):
            img_t, cond_t, rast_t, cheat, base = val_dataset[idx]
            img = img_t.unsqueeze(0).to(device)
            cond = cond_t.unsqueeze(0).to(device)
            rast = rast_t.unsqueeze(0).to(device)
            pred = model(img, cond=cond)
            gt_01   = (rast * 0.5 + 0.5).clamp(0,1)
            pred_01 = (pred * 0.5 + 0.5).clamp(0,1)
            out_name = f"epoch{epoch+1:03d}_{i:02d}_{base}.png"
            _save_side_by_side_png(gt_01, pred_01, Path(out_dir)/out_name)


def train(cfg: Cfg):
    local_rank, world_size, rank = init_distributed()
    set_seed(cfg.seed + rank)

    set_seed(cfg.seed)

    # Datasets / loaders
    train_ds = CustomDataset(cfg.train_img_dir, cfg.train_rast_dir, '/work/wildfirerisk/vlm_forward_pass_predictions_small_dataset.json')
    val_ds   = CustomDataset(cfg.val_img_dir, cfg.val_rast_dir, '/work/wildfirerisk/vlm_forward_pass_predictions_small_dataset.json')
    train_ds = CustomDataset(cfg.train_img_dir, cfg.train_rast_dir, cfg.climate_json)
    val_ds   = CustomDataset(cfg.val_img_dir, cfg.val_rast_dir, cfg.climate_json)

    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed); random.seed(seed)

    # Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False)  if dist.is_initialized() else None

    def _worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed); random.seed(seed)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )


    # Build model exactly per instructions
    model, device = build_model_and_device(local_rank)
    if dist.is_available() and dist.is_initialized():
        # Only parameters with requires_grad=True (fusion) will get grads synced
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)


    # Optimizer on unfrozen params only (fusion)
    base_model = model.module if isinstance(model, DDP) else model
    params = [p for p in base_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=3e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    ssim = SSIM().to(device)
    huber = nn.SmoothL1Loss(beta=cfg.huber_delta)

    os.makedirs(os.path.join(cfg.out_dir, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, 'val_samples'), exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Fixed validation sample indices (same 40 every epoch)
    k = min(40, len(val_ds))
    rng = random.Random(12345)
    fixed_indices = rng.sample(list(range(len(val_ds))), k=k) if is_main_process() else []


    best_val = float('inf')
    global_step = 0
    with open(os.path.join(cfg.out_dir, 'logs.out'), 'w') as f:
        for epoch in range(cfg.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            run_recon = run_ssim = run_grad = 0.0
            t0 = time.time()

            opt.zero_grad(set_to_none=True)  # <-- zero once at epoch start

            for it, (img, climate, rast, cheat, base) in enumerate(train_dl):
                img = img.to(device, non_blocking=True)
                cond = climate.to(device, non_blocking=True)
                rast = rast.to(device, non_blocking=True)

                pred = model(img, cond=cond)
                l_recon = huber(pred, rast)
                pred_01 = (pred * 0.5 + 0.5).clamp(0,1)
                rast_01 = (rast * 0.5 + 0.5).clamp(0,1)
                l_ssim = 1 - ssim(pred_01, rast_01)
                l_grad = grad_loss(pred, rast)
                loss = cfg.w_recon*l_recon + cfg.w_ssim*l_ssim + cfg.w_grad*l_grad

                # --- accumulate ---
                loss = loss / cfg.accum_steps
                scaler.scale(loss).backward()

                run_recon += l_recon.item()
                run_ssim  += l_ssim.item()
                run_grad  += l_grad.item()

                do_step = ((it + 1) % cfg.accum_steps == 0) or (it + 1 == len(train_dl))
                if do_step:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(params, cfg.grad_clip)

                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1  # counts optimizer updates

                if is_main_process() and (it+1) % cfg.log_every == 0:
                    nlog = cfg.log_every
                    print(f"[epoch {epoch+1:03d}] it {it+1:04d}/{len(train_dl)} "
                        f"recon={run_recon/nlog:.4f} ssim={run_ssim/nlog:.4f} grad={run_grad/nlog:.4f}", file=f, flush=True)
                    run_recon = run_ssim = run_grad = 0.0
            if epoch % cfg.eval_every == 0:
                val_metrics = evaluate(model, device, val_dl, ssim, huber, cfg)
                print(f"[val @ epoch {epoch+1:03d}] recon={val_metrics['recon']:.4f} ssim={val_metrics['ssim']:.4f} grad={val_metrics['grad']:.4f}", file=f, flush=True)

                if is_main_process():
                    ckpt_path = os.path.join(cfg.out_dir, 'ckpts', f"fusion_epoch{epoch:03d}.pt")

                    if val_metrics['recon'] < best_val:
                        save_val_samples_fixed(base_model, device, val_ds, fixed_indices, os.path.join(cfg.out_dir, 'val_samples'), epoch)
                        best_val = val_metrics['recon']
                        torch.save({'epoch': epoch, 'model_state': base_model.state_dict(), 'metrics': val_metrics},
                                ckpt_path)



            scheduler.step()


    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_main_process():
        print("Training complete.")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    # print("Training complete.")


# ---------------------------
# CLI
# ---------------------------

def build_cfg_from_args():
    import argparse
    p = argparse.ArgumentParser("Train fusion on satellite->raster with climate conditioning")
    p.add_argument('--train_img_dir', type=str, default='/work/wildfirerisk/small_dataset/alphaearth_embeddings/train')
    p.add_argument('--train_rast_dir', type=str, default='/work/wildfirerisk/small_dataset/normalised_risk_rasters/train')
    p.add_argument('--val_img_dir',   type=str, default='/work/wildfirerisk/small_dataset/alphaearth_embeddings/train')
    p.add_argument('--val_rast_dir',  type=str, default='/work/wildfirerisk/small_dataset/normalised_risk_rasters/train')
    p.add_argument('--climate_json',  type=str, default='/work/wildfirerisk/small_dataset/climate_data.json')
    p.add_argument('--out_dir', type=str, default='/work/wildfirerisk/trainings/small/alphaearth_oracle_no_cot')
    p.add_argument('--eval_every', type=int, default=25)
    p.add_argument('--accum_steps', type=int, default=1)

    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--w_recon', type=float, default=1.0)
    p.add_argument('--w_ssim', type=float, default=0.5)
    p.add_argument('--w_grad', type=float, default=0.2)
    p.add_argument('--huber_delta', type=float, default=1.0)

    args = p.parse_args()
    return Cfg(
        train_img_dir=args.train_img_dir,
        train_rast_dir=args.train_rast_dir,
        val_img_dir=args.val_img_dir,
        val_rast_dir=args.val_rast_dir,
        climate_json=args.climate_json,
        out_dir=args.out_dir,
        eval_every=args.eval_every,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        seed=args.seed,
        w_recon=args.w_recon,
        w_ssim=args.w_ssim,
        w_grad=args.w_grad,
        huber_delta=args.huber_delta,
    )


if __name__ == '__main__':
    cfg = build_cfg_from_args()
    train(cfg)