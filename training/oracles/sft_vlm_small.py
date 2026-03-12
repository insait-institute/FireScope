# sft_vlm_digits_ddp.py
import os
import sys
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# -------------------------
# Project setup (same as yours)
# -------------------------
project_dir = f'/home/{os.environ["USER"]}/FireScope'
os.chdir(project_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from config import DATA_DIR
from prompts import sft_prompt as main_prompt

import numpy as np
from PIL import Image as PILImage

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from transformers import Qwen2_5_VLProcessor, AutoModelForImageTextToText

from functools import partial

def batch_metrics(next_token_logits, targets, tokenizer):
    preds = next_token_logits.argmax(dim=-1)  # [B]
    pred_digits, true_digits = [], []
    for pid, tid in zip(preds.tolist(), targets.tolist()):
        pred_str = tokenizer.decode([pid]).strip()
        true_str = tokenizer.decode([tid]).strip()
        # Convert to ints if possible
        try:
            p = int(pred_str)
        except ValueError:
            p = None
        try:
            t = int(true_str)
        except ValueError:
            t = None

        # Assign penalty for invalid preds or labels
        if p is None or not (0 <= p <= 9) or t is None or not (0 <= t <= 9):
            mae_i = 9.0
            mse_i = 81.0
            acc_i = 0.0
        else:
            diff = abs(p - t)
            mae_i = diff
            mse_i = diff ** 2
            acc_i = 1.0 if p == t else 0.0

        pred_digits.append(p if p is not None else -1)
        true_digits.append(t if t is not None else -1)
        # accumulate per-sample metrics
        if 'maes' not in locals():
            maes, mses, accs = [], [], []
        maes.append(mae_i)
        mses.append(mse_i)
        accs.append(acc_i)

    mae = float(np.mean(maes)) if maes else np.nan
    mse = float(np.mean(mses)) if mses else np.nan
    acc = float(np.mean(accs)) if accs else np.nan
    return mae, mse, acc


def make_loader(ds, sampler, batch_size, shuffle, processor):
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # reuse workers
        collate_fn=partial(collate_fn, processor=processor),  # <- no lambda
    )
# -------------------------
# Data logic identical to your GRPO setup
# - We reuse the exact splitting/labeling rules
# - For SFT we train on 2x2 tiles like your train_data
# - For eval we keep full tiles (not strictly required; left simple)
# -------------------------
def make_dp(data_dict, eval=False, img_root=f'{DATA_DIR}/small_dataset/satellite_images/'):
    raster_root = f'{DATA_DIR}/small_dataset/normalised_risk_rasters/'
    name = data_dict['tile_file']
    image_path = f"{img_root}/{name.replace('npy', 'png')}"
    raster_path = os.path.join(raster_root, name.replace('png', 'npy'))
    input_text = main_prompt.build_prompt(data_dict['climate'])

    messages = [{"role": "user", "content": input_text}]

    if eval:
        return {
            "prompt_messages": messages,
            "image_path": image_path,
            "crop_box": None,  # full image
            "solution": data_dict['label'],
        }

    # Non-eval: 2x2 tiles
    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size
    mid_w, mid_h = W // 2, H // 2
    crop_boxes = [
        (0,      0,      mid_w, mid_h),  # TL
        (mid_w,  0,      W,     mid_h),  # TR
        (0,      mid_h,  mid_w, H),      # BL
        (mid_w,  mid_h,  W,     H),      # BR
    ]

    raster = np.load(raster_path)
    if raster.ndim != 2:
        raise ValueError(f"Expected 2D raster (H x W), got {raster.shape}")
    rH, rW = raster.shape
    rmid_w, rmid_h = rW // 2, rH // 2
    raster_tiles = [
        raster[0:rmid_h,       0:rmid_w],     # TL
        raster[0:rmid_h,       rmid_w:rW],    # TR
        raster[rmid_h:rH,      0:rmid_w],     # BL
        raster[rmid_h:rH,      rmid_w:rW],    # BR
    ]
    raster_means = [float(np.mean(t)) for t in raster_tiles]
    solutions = [min(int(m * 10), 9) for m in raster_means]

    dps = []
    for cb, sol in zip(crop_boxes, solutions):
        dps.append({
            "prompt_messages": messages,
            "image_path": image_path,
            "crop_box": cb,
            "solution": sol,
        })
    return dps


# -------------------------
# Precompute: chat-templated prompts -> tokenized input_ids (CPU)
# Also precompute the single target token id for the digit.
# -------------------------
@dataclass
class PrecomputedItem:
    chat_text: str                 # chat-templated string (assistant turn open)
    image_path: str
    crop_box: Tuple[int, int, int, int] | None
    target_id: int                 # single token id for the digit



class VLMDigitsDataset(Dataset):
    def __init__(self, items: List[PrecomputedItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, processor: Qwen2_5_VLProcessor):
    # Prepare images
    images = []
    for b in batch:
        img = PILImage.open(b.image_path).convert("RGB")
        if b.crop_box is not None:
            img = img.crop(b.crop_box)
        images.append(img)

    # Prepare texts (chat-templated strings)
    texts = [b.chat_text for b in batch]

    # Let the processor build everything consistently
    enc = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    # enc has: input_ids, attention_mask, pixel_values, image_grid_thw
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    pixel_values = enc["pixel_values"]
    image_grid_thw = enc.get("image_grid_thw")

    # position where the next token (first assistant token) is predicted
    # (sum over attention mask - 1) for each sample
    last_index = attention_mask.sum(dim=1) - 1  # [B]

    targets = torch.tensor([b.target_id for b in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "last_index": last_index,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "targets": targets,
    }




# -------------------------
# Training (DDP-friendly) – Cross-Entropy on the very next token only
# -------------------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main():
    # ---------------------
    # Setup / args (minimal)
    # ---------------------
    output_dir = f"{DATA_DIR}/trainings/sft_digits_small"
    os.makedirs(output_dir, exist_ok=True)

    use_ddp, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---------------------
    # Model & Processor
    # ---------------------
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        # safety: set pad token to eos if undefined
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()  # saves a lot of VRAM on Qwen-VL
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ---------------------
    # Build train/val metadata (same as your GRPO code)
    # ---------------------
    with open(f'/work/wildfirerisk/small_dataset/risk_rasters/tile_extrametadata.json', 'r') as f:
        md = json.load(f)
    targets = os.listdir(f'{DATA_DIR}/small_dataset/satellite_images/train') + os.listdir(f'{DATA_DIR}/small_dataset/satellite_images/val')
    md = [x for x in md if x['tile_file'].split('/')[-1].replace('npy', 'png') in targets]
    with open(f'/work/wildfirerisk/small_dataset/climate_data.json', 'r') as f:
        cd = json.load(f)

    train_md = {f"{round(m['centroid_lat'], 4)}_{round(m['centroid_lon'], 4)}": (min(int(m['mean_normalised_risk']*10), 9), m['tile_file']) for m in md if m['subset'] == 'train'}
    val_md   = {f"{round(m['centroid_lat'], 4)}_{round(m['centroid_lon'], 4)}": (min(int(m['mean_normalised_risk']*10), 9), m['tile_file']) for m in md if m['subset'] == 'val'}

    trainset_rows = [{'label': train_md[k][0], 'climate': cd[k], 'tile_file': train_md[k][1]} for k in train_md]
    valset_rows   = [{'label': val_md[k][0],   'climate': cd[k], 'tile_file': val_md[k][1]}   for k in val_md]

    # ---------------------
    # Precompute (CPU): chat-templated prompts -> input_ids; target digit -> target_id
    # We assert the digit class tokenizes to a single token.
    # ---------------------
    def precompute(rows, eval_mode=False) -> List[PrecomputedItem]:
        items: List[PrecomputedItem] = []
        for rd in rows:
            for dp in (make_dp(rd, eval=eval_mode) if not eval_mode else [make_dp(rd, eval=True)]):
                input_text = dp["prompt_messages"][0]["content"]
                messages_for_template = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": input_text}
                    ]
                }]

                chat_text = processor.apply_chat_template(
                    messages_for_template,
                    add_generation_prompt=True,
                    tokenize=False   # <-- return a string with image placeholders & assistant turn
                )

                # target digit token (unchanged)
                target_digit = str(int(dp["solution"]))
                tgt_ids = tokenizer(target_digit, add_special_tokens=False)["input_ids"]
                if len(tgt_ids) != 1:
                    raise ValueError(f"Digit '{target_digit}' tokenized into {len(tgt_ids)} tokens: {tgt_ids}")
                target_id = tgt_ids[0]

                items.append(PrecomputedItem(
                    chat_text=chat_text,
                    image_path=dp["image_path"],
                    crop_box=dp["crop_box"],
                    target_id=target_id
                ))

        return items

    if is_main_process():
        print("Precomputing train items...")
    train_items = precompute(trainset_rows, eval_mode=True)
    # (Optional) tiny val set; not needed for core training loop
    if is_main_process():
        print("Precomputing eval items...")
    val_items = precompute(valset_rows, eval_mode=True)  # full tiles

    train_dataset = VLMDigitsDataset(train_items)
    val_dataset = VLMDigitsDataset(val_items)

    # ---------------------
    # Samplers / Loaders (DDP-aware)
    # ---------------------
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None


    train_loader = make_loader(train_dataset, train_sampler, batch_size=4, shuffle=True, processor=processor)
    val_loader   = make_loader(val_dataset,   val_sampler,   batch_size=4, shuffle=False, processor=processor)

    # ---------------------
    # Optimizer (minimal)
    # ---------------------
    lr = 1e-5
    optim = torch.optim.AdamW(model.parameters(), lr=lr, fused=True if torch.cuda.is_available() else False)

    # ---------------------
    # Train loop (only the “first next token” CE)
    # ---------------------
    num_epochs = 10000

    def step_batch(batch):
        # Move tensors to device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True, dtype=model.module.dtype if use_ddp else model.dtype)
        last_index = batch["last_index"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        # Forward
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            # The model returns logits over text vocab for each position in input_ids
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=batch["image_grid_thw"],  # <-- required by Qwen2.5-VL
                use_cache=False,
            )
            B, T, V = out.logits.shape
            gather_index = last_index.view(B, 1, 1).expand(B, 1, V)
            next_token_logits = out.logits.gather(dim=1, index=gather_index).squeeze(1)
            loss = torch.nn.functional.cross_entropy(next_token_logits, targets)


        return loss
    accum_steps = 4
    eval_every = 1
    save_every = 10
    best_mae = 100
    for epoch in range(num_epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            loss = step_batch(batch) / accum_steps   # scale for accumulation
            loss.backward()

            # step every accum_steps or on the last micro-batch
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                # optional: clip to avoid inf grads
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum_steps  # unscale for reporting

            # log every 50 *accumulated* steps
            if is_main_process() and ((i + 1) % (50 * accum_steps) == 0):
                avg_loss = running_loss / (50 * accum_steps)
                print(f"Epoch {epoch+1} Step {i+1} | Avg Loss {avg_loss:.4f}")
                running_loss = 0.0

        if is_main_process():
            denom = max(1, len(train_loader))
            print(f"Epoch {epoch+1} | Avg Train Loss: {running_loss / denom:.4f}")

        if (epoch + 1) % eval_every == 0:
            model.eval()
            to_save = False
            with torch.no_grad():
                val_loss = 0.0
                val_mae, val_mse, val_acc = 0.0, 0.0, 0.0
                val_count = 0
                for batch in val_loader:
                    loss = step_batch(batch)
                    val_loss += loss.item()

                    # ---- Compute metrics ----
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    pixel_values = batch["pixel_values"].to(
                        device, non_blocking=True, dtype=model.module.dtype if use_ddp else model.dtype
                    )
                    last_index = batch["last_index"].to(device, non_blocking=True)
                    targets = batch["targets"].to(device, non_blocking=True)

                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=batch["image_grid_thw"],
                        use_cache=False,
                    )
                    B, T, V = out.logits.shape
                    gather_index = last_index.view(B, 1, 1).expand(B, 1, V)
                    next_token_logits = out.logits.gather(dim=1, index=gather_index).squeeze(1)

                    mae, mse, acc = batch_metrics(next_token_logits, targets, processor.tokenizer)
                    val_mae += mae
                    val_mse += mse
                    val_acc += acc
                    val_count += 1
                if val_mae/val_count < best_mae:
                    best_mae = val_mae/val_count
                    to_save = True
                if is_main_process() and val_count > 0:
                    print(
                        f"Epoch {epoch+1} | Val Loss: {val_loss/val_count:.4f} | "
                        f"MAE: {val_mae/val_count:.4f} | MSE: {val_mse/val_count:.4f} | Acc: {val_acc/val_count:.4f}"
                    )


        # Save only on main process
        if is_main_process() and ((epoch+1)%save_every == 0 or to_save):
            if to_save:
                name = 'best.pt'
            else:
                name = f"epoch_{epoch+1}.pt"
            save_path = os.path.join(output_dir, name)
            to_save = model.module if use_ddp else model
            to_save.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"Saved to {save_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
