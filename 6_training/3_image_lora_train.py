"""
Z-Image LoRA 학습 스크립트
모델: Tongyi-MAI/Z-Image
데이터: mks0813/pixel_image_dataset
"""

import torch
from datasets import load_dataset
from diffusers import ZImagePipeline
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# 설정
MODEL_ID = "Tongyi-MAI/Z-Image"
DATASET_ID = "mks0813/pixel_image_dataset"
OUTPUT_DIR = "./outputs/z-image-pixel-lora"

EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LORA_RANK = 32
LORA_ALPHA = 64
RESOLUTION = 1024
GRADIENT_ACCUMULATION_STEPS = 4


def collate_fn(examples):
    images = [example["image"].convert("RGB").resize((RESOLUTION, RESOLUTION)) for example in examples]
    prompts = [example["prompt"] for example in examples]
    return {"images": images, "prompts": prompts}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 파이프라인 로드
    print("모델 로딩...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    # LoRA 설정
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )

    # Transformer에 LoRA 적용
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()
    pipe.to(device)

    # 데이터셋 로드
    print("데이터셋 로딩...")
    dataset = load_dataset(DATASET_ID, split="train")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=LEARNING_RATE)

    # 학습 루프
    print("학습 시작...")
    pipe.transformer.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(progress_bar):
            # 이미지 인코딩
            images = batch["images"]
            prompts = batch["prompts"]

            # Forward pass (diffusers 내부 학습 로직 활용)
            loss = pipe.train_step(
                prompt=prompts,
                image=images,
            )

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 평균 Loss: {avg_loss:.4f}")

        # 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            save_path = f"{OUTPUT_DIR}/checkpoint-{epoch + 1}"
            pipe.transformer.save_pretrained(save_path)
            print(f"체크포인트 저장: {save_path}")

    # 최종 저장
    pipe.transformer.save_pretrained(OUTPUT_DIR)
    print(f"학습 완료! 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
