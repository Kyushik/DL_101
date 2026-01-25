"""
CLIP 텍스트 인코더를 활용한 이미지 검색
"""

import torch
import faiss
import matplotlib.pyplot as plt
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


# 2. FAISS 인덱스 및 데이터셋 로드
print("FAISS 인덱스 로딩 중...")
index = faiss.read_index("5_huggingface/image_db/caltech256.index")

print("데이터셋 로딩 중...")
dataset = load_dataset("bitmind/caltech-256", split="train")


# 3. 검색 함수
def search_images(query, top_k=3):
    """텍스트 쿼리로 유사한 이미지 검색"""
    # 텍스트 임베딩
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
        text_embeds = text_embeds / get_vector_norm(text_embeds)  # 정규화

    # FAISS 검색
    query_vector = text_embeds.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vector, top_k)

    return scores[0], indices[0]


# 4. 결과 시각화 함수
def show_results(query, scores, indices):
    """검색 결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Search: "{query}"', fontsize=14)

    for i, (score, idx) in enumerate(zip(scores, indices)):
        image = dataset[int(idx)]["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        axes[i].imshow(image)
        axes[i].set_title(f"#{i+1} (score: {score:.3f})")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# 5. 검색 실행
if __name__ == "__main__":
    query = input("\n검색어를 입력하세요 (영어): ")

    print(f"\n'{query}' 검색 중...")
    scores, indices = search_images(query, top_k=3)

    print("\n검색 결과:")
    for i, (score, idx) in enumerate(zip(scores, indices)):
        print(f"  {i+1}. index {idx} (score: {score:.3f})")

    show_results(query, scores, indices)
