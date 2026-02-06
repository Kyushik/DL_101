"""
Z-Image LoRA 학습 (DiffSynth Studio)

설치: pip install diffsynth
실행: python 3_image_lora_train.py

※ DiffSynth-Studio clone 필요:
   git clone https://github.com/modelscope/DiffSynth-Studio.git
"""

import subprocess
import csv
import os
from pathlib import Path
from datasets import load_dataset

os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

# 경로 설정
DIFFSYNTH_DIR = "./DiffSynth-Studio"
DATASET_PATH = "./data/pixel_dataset"
OUTPUT_DIR = "./outputs/z-image-pixel-lora"

# 모델 설정
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
MODEL_FILES = "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/*.safetensors"
DATASET_ID = "mks0813/pixel_image_dataset"

# 학습 설정
EPOCHS = 3
LEARNING_RATE = 1e-4
LORA_RANK = 32
LORA_TARGET_MODULES = "to_q,to_k,to_v,to_out.0,w1,w2,w3"


def prepare_dataset():
    """HuggingFace 데이터셋을 DiffSynth 형식으로 변환"""
    from tqdm import tqdm

    print("데이터셋 준비 중...")
    img_dir = f"{DATASET_PATH}/images"
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID, split="train")

    metadata = []
    for i, item in enumerate(tqdm(dataset, desc="이미지 저장")):
        img_path = f"images/{i:05d}.png"
        item["image"].convert("RGB").save(f"{DATASET_PATH}/{img_path}")
        prompt = item.get("prompt", item.get("text", "pixel art"))
        metadata.append({"image": img_path, "prompt": prompt})

    csv_path = f"{DATASET_PATH}/metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "prompt"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"완료: {len(metadata)}개 이미지 -> {csv_path}")


def train():
    """DiffSynth Studio로 LoRA 학습"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 데이터셋 준비
    csv_path = f"{DATASET_PATH}/metadata.csv"
    if not Path(csv_path).exists():
        prepare_dataset()

    # DiffSynth-Studio 디렉토리로 이동하여 실행
    train_script = f"{DIFFSYNTH_DIR}/examples/z_image/model_training/train.py"

    cmd = [
        "accelerate", "launch", train_script,
        "--model_id_with_origin_paths", MODEL_FILES,
        "--dataset_base_path", DATASET_PATH,
        "--dataset_metadata_path", csv_path,
        "--dataset_repeat", "1",
        "--max_pixels", "1048576",
        "--trainable_models", "dit",
        "--learning_rate", f"{LEARNING_RATE}",
        "--num_epochs", f"{EPOCHS}",
        "--lora_base_model", "dit",
        "--lora_target_modules", LORA_TARGET_MODULES,
        "--lora_rank", f"{LORA_RANK}",
        "--output_path", OUTPUT_DIR,
        "--use_gradient_checkpointing",
    ]

    print(f"학습 명령어:\n{' '.join(cmd)}")
    print(f"\n학습 시작...")
    subprocess.run(cmd)


if __name__ == "__main__":
    train()
