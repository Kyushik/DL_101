"""
CLIP(텍스트 이해) + FAISS(빠른 벡터 검색)로
"텍스트로 이미지 찾기"를 하는 코드

필요한 것:
- 미리 만들어둔 FAISS 인덱스 파일: 5_huggingface/image_db/caltech256.index
- Caltech-256 데이터셋(약 3만 장 이미지)
"""

import torch
import faiss
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F


# -----------------------------
# 1) CLIP 모델 준비 (텍스트를 임베딩 벡터로 바꾸는 모델)
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
# 2) FAISS 인덱스 + 데이터셋 로드
# -----------------------------
print("\nFAISS 인덱스 로딩 중...")
index_path = "5_huggingface/image_db/caltech256.index"
index = faiss.read_index(index_path)
print(f"인덱스 로드 완료: {index_path}")

print("\n데이터셋 로딩 중...")
dataset = load_dataset("bitmind/caltech-256", split="train")
print(f"이미지 개수: {len(dataset)}")


# -----------------------------
# 3) 텍스트로 이미지 검색하는 함수
# -----------------------------
def search_images_by_text(query_text: str, top_k: int = 3):
    """
    입력한 영어 문장(query_text)을 "벡터(숫자 묶음)"로 바꾼 다음,
    FAISS에서 가장 비슷한 이미지 top_k개를 찾아준다.
    """
    # (1) 텍스트를 CLIP이 처리 가능한 형태로 변환
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # (2) CLIP으로 텍스트 임베딩(벡터) 뽑기
    with torch.no_grad():
        text_vector = model.get_text_features(**inputs)

        # 벡터 길이를 1로 맞춤(정규화)
        text_vector = F.normalize(text_vector, p=2, dim=-1)

    # (3) FAISS 검색 (가장 비슷한 이미지의 번호와 점수 반환)
    query_vector = text_vector.cpu().numpy().astype("float32")
    scores, indices = index.search(query_vector, top_k)

    # scores, indices는 (1, top_k) 모양이라 0번째만 꺼내서 top_k 개를 반환
    return scores[0], indices[0]


# -----------------------------
# 4) 검색 결과를 화면에 보여주는 함수
# -----------------------------
def show_search_results(query_text: str, scores, indices):
    """
    검색된 이미지들을 화면에 출력한다.
    scores: 비슷한 정도 점수(높을수록 더 비슷)
    indices: 데이터셋에서의 이미지 번호
    """
    top_k = len(indices)

    fig, axes = plt.subplots(1, top_k, figsize=(4 * top_k, 4))
    fig.suptitle(f'Search: "{query_text}"', fontsize=14)

    # top_k가 1인 경우 axes가 배열이 되도록 처리
    if top_k == 1:
        axes = [axes]

    for i, (score, idx) in enumerate(zip(scores, indices)):
        image = dataset[int(idx)]["image"]

        # 이미지가 RGB가 아니면 RGB로 바꿔서 출력
        if image.mode != "RGB":
            image = image.convert("RGB")

        axes[i].imshow(image)
        axes[i].set_title(f"#{i+1}\nscore: {score:.3f}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 5) 실제 실행 부분
# -----------------------------
if __name__ == "__main__":
    query = input("\n검색어를 입력하세요 (영어, 예: 'a cute dog'): ").strip()
    if not query:
        print("검색어가 비어있어요. 프로그램을 종료합니다.")
        raise SystemExit

    top_k = 3  # 보고 싶은 결과 개수
    print(f"\n'{query}' 검색 중...")

    scores, indices = search_images_by_text(query, top_k=top_k)

    print("\n검색 결과(숫자가 클수록 더 비슷):")
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        print(f"  {rank}. 데이터셋 번호: {int(idx)} / score: {score:.3f}")

    show_search_results(query, scores, indices)
