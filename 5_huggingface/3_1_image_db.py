"""
CLIP(이미지 인코더) + FAISS(벡터 검색 라이브러리)로
이미지들을 "벡터 DB"로 만들어 저장하는 코드

데이터셋: Caltech-256 (약 3만 장 이미지)
결과물: caltech256.index (FAISS 인덱스 파일)
"""

import os
import torch
import faiss
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F  


# -----------------------------
# 1) CLIP 모델 준비
# -----------------------------
print("CLIP 모델 로딩 중...")
model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # 학습이 아니라 추론 모드로 설정
print(f"디바이스: {device}")


# -----------------------------
# 2) 데이터셋 로드
# -----------------------------
print("\nCaltech-256 데이터셋 다운로드 중...")
dataset = load_dataset("bitmind/caltech-256", split="train")
print(f"이미지 개수: {len(dataset)}")


# -----------------------------
# 3) 이미지 → 임베딩(벡터) 만들기
# -----------------------------
print("\n이미지 임베딩 생성 중...")
batch_size = 32
all_embeddings = []

for start in range(0, len(dataset), batch_size):
    batch = dataset[start : start + batch_size]

    # RGB 이미지가 아니면 RGB로 바꿔줌
    images = [
        (img.convert("RGB") if img.mode != "RGB" else img)
        for img in batch["image"]
    ]

    # 이미지를 모델 입력 형태(텐서)로 변환
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 특징(임베딩) 추출: 학습이 아니라 추론만 하므로 no_grad 사용
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)

        # 벡터 길이를 1로 맞춤(정규화)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

    # 이미지 임베딩을 numpy로 변경 후 리스트에 추가
    all_embeddings.append(image_embeds.cpu().numpy())

    # 진행 과정 표시
    if (start // batch_size + 1) % 50 == 0:
        done = min(start + batch_size, len(dataset))
        print(f"  진행: {done}/{len(dataset)}")

# (N, 512) 형태로 합치기
embeddings = np.vstack(all_embeddings).astype("float32")
print(f"\n임베딩 shape: {embeddings.shape}")


# -----------------------------
# 4) FAISS 인덱스 만들고 저장하기
# -----------------------------
print("\nFAISS 인덱스 생성 중...")

# CLIP 임베딩 차원(보통 512)
dimension = embeddings.shape[1]

# IndexFlatIP: 벡터끼리 얼마나 비슷한지 비교하고 벡터들을 정리
# 인덱스 사용: 인덱스 = 빠른 검색을 위해 미리 정리한 목록
# 예시: 책에서 "사과"가 나오는 곳을 찾을 때 페이지를 알면 빨리 찾음
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"인덱스에 저장된 벡터 수: {index.ntotal}")

# 저장 폴더 만들고 인덱스 파일 저장
save_dir = "5_huggingface/image_db"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "caltech256.index")
faiss.write_index(index, save_path)

print(f"\n저장 완료: {save_path}")
