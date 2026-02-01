"""
MBTI F/T Style 데이터셋 처리 스크립트
mks0813/mbti-f-t-style-responses 데이터를 axolotl 학습 형식으로 변환
"""

from datasets import load_dataset
import json
from pathlib import Path


def process_mbti_dataset():
    """
    Hugging Face에서 MBTI 데이터셋을 로드하고
    axolotl 학습에 맞는 형식으로 변환합니다.
    """
    # 데이터셋 로드
    print("데이터셋 로딩 중...")
    dataset = load_dataset("mks0813/mbti-f-t-style-responses")

    # train split 가져오기
    train_data = dataset["train"]

    print(f"전체 데이터 수: {len(train_data)}")
    print(f"컬럼: {train_data.column_names}")

    # 샘플 데이터 확인
    print("\n=== 샘플 데이터 ===")
    print(train_data[0])

    # axolotl 형식으로 변환 (conversations 형식)
    processed_data = []

    for item in train_data:
        conversation = item.get("conversation", "")
        f_style_response = item.get("f_style_response", "")

        # 빈 데이터 스킵
        if not conversation or not f_style_response:
            continue

        # Qwen3 chat 형식에 맞게 변환
        processed_item = {
            "messages": [
                {
                    "role": "user",
                    "content": conversation
                },
                {
                    "role": "assistant",
                    "content": f_style_response
                }
            ]
        }
        processed_data.append(processed_item)

    print(f"\n처리된 데이터 수: {len(processed_data)}")

    # 저장 경로 설정
    output_dir = "./data"
    output_dir.mkdir(exist_ok=True)

    # JSONL 형식으로 저장 (검증 데이터는 axolotl에서 자동 분리)
    train_path = output_dir / "mbti_f_style_train.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n저장 완료!")
    print(f"학습 데이터: {train_path}")

    # 샘플 출력
    print("\n=== 변환된 샘플 데이터 ===")
    print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))

    return train_path


if __name__ == "__main__":
    process_mbti_dataset()
