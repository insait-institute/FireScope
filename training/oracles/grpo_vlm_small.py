import sys
import os
project_dir = f'/home/{os.environ["USER"]}/FireScope'
os.chdir(project_dir)
# Add to sys.path so Python can find your modules
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
from config import DATA_DIR
import torch
from datasets import Dataset, Features, Value, Image
from prompts import main_prompt
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)
from PIL import Image as PILImage
import json
import rewards.common as rewards
from collections import defaultdict
from transformers import Qwen2_5_VLProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model


def make_dp(data_dict, img_root=f'/work/wildfirerisk/small_dataset/satellite_images/'):
    name = data_dict['tile_file']
    image_path = f"{img_root}/{name.replace('npy', 'png')}"
    input_text = main_prompt.build_prompt(data_dict['climate'])

    messages = [
        {
            "role": "user",
            "content": input_text,
        }
    ]
    return {
        "prompt": messages,
        "image": PILImage.open(image_path).convert("RGB"),
        "solution": data_dict['label']
    }

if __name__ == "__main__":
    ################
    # Model
    ################
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id, use_fast=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    ################
    # Dataset
    ################
    import json
    with open(f'{DATA_DIR}/small_dataset/risk_rasters/tile_extrametadata.json', 'r') as f:
        md = json.load(f)
    with open(f'{DATA_DIR}/small_dataset/climate_data.json', 'r') as f:
        cd = json.load(f)
    train_md = {f"{m['centroid_lat']}_{m['centroid_lon']}": (min(int(m['mean_normalised_risk']*10), 9), m['tile_file']) for m in md if m['subset'] == 'train'}
    val_md = {f"{m['centroid_lat']}_{m['centroid_lon']}": (min(int(m['mean_normalised_risk']*10), 9), m['tile_file']) for m in md if m['subset'] == 'val'}
    trainset = [{'label': train_md[k][0], 'climate': cd[k], 'tile_file': train_md[k][1]} for k in train_md]
    valset = [{'label': val_md[k][0], 'climate': cd[k], 'tile_file': val_md[k][1]} for k in val_md]
    frequencies = defaultdict(int) # frequencies for reward weights
    for rd in trainset:
        frequencies[rd['label']] += 1
    train_data = [make_dp(rd)
    for rd in trainset
    ]
    eval_data = [make_dp(rd)
    for rd in valset
    ]
    features = Features({
        "prompt": [{
            "role": Value("string"),
            "content": Value("string")
        }],
        "image": Image(),  # Base64-encoded
        "solution": Value("int32"),
    })
    train_dataset = Dataset.from_list(train_data, features=features)
    eval_dataset = Dataset.from_list(eval_data, features=features)
    ################
    # Training
    ################
    config = GRPOConfig(
        output_dir=f"{DATA_DIR}/trainings/grpo_1",
        learning_rate=1e-5,
        num_train_epochs=100,
        num_generations=4,  # Number of completions to generate for each prompt
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_prompt_length=4096,
        max_completion_length=1024,
        bf16=True,
        logging_steps=5,
        eval_steps=100,
        eval_strategy='steps',
        save_steps=100,
        log_completions=True,
        disable_tqdm=False,
        logging_strategy='steps',
        logging_dir=f"{DATA_DIR}/trainings/grpo_1",
        reward_weights=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        run_name='vlm_full',
        beta=0.01,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=config,
        reward_funcs=[rewards.generate_accuracy_reward(frequencies), rewards.is_final_answer_format_reward, rewards.absolute_error, rewards.squared_error, rewards.final_answer, rewards.ground_truth],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()