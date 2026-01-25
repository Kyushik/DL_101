"""
CLIP 이미지 인코더 + FAISS를 활용한 이미지 DB 구축
데이터셋: Caltech-256 (256개 다양한 카테고리, 30,607개 이미지)
"""

import os
import torch
import faiss
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel


def get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """L2 정규화를 위한 벡터 norm 계산"""
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


# 1. CLIP 모델 로드
print("CLIP 모델 로딩 중...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"디바이스: {device}")


# 2. Caltech-256 데이터셋 로드 (Hugging Face)
print("\nCaltech-256 데이터셋 다운로드 중... (처음엔 시간이 걸립니다)")
dataset = load_dataset("bitmind/caltech-256", split="train")
print(f"이미지 개수: {len(dataset)}")

# 3. 이미지 임베딩 생성
print("\n이미지 임베딩 생성 중...")
batch_size = 32
all_embeddings = []

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i + batch_size]
    images = []

    for img in batch["image"]:
        # 그레이스케일 -> RGB 변환
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    # CLIP 전처리 및 임베딩
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        image_embeds = image_embeds / get_vector_norm(image_embeds)  # 정규화

    all_embeddings.append(image_embeds.cpu().numpy())

    if (i // batch_size + 1) % 50 == 0:
        print(f"  진행: {min(i + batch_size, len(dataset))}/{len(dataset)}")

embeddings = np.vstack(all_embeddings).astype("float32")
print(f"\n임베딩 shape: {embeddings.shape}")


# 4. FAISS 인덱스 생성 및 저장
print("\nFAISS 인덱스 생성 중...")
dimension = embeddings.shape[1]  # 512
index = faiss.IndexFlatIP(dimension)  # 내적(코사인 유사도) 기반
index.add(embeddings)
print(f"인덱스에 저장된 벡터 수: {index.ntotal}")

# 저장
save_dir = "5_huggingface/image_db"
os.makedirs(save_dir, exist_ok=True)

faiss.write_index(index, f"{save_dir}/caltech256.index")

print(f"\n저장 완료: {save_dir}/")
print("  - caltech256.index (FAISS 인덱스)")
