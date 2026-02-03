"""
Axolotl 학습 모델 추론 스크립트
Full fine-tuning 및 LoRA 모델 모두 지원
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def load_model(model_path, base_model=None, is_lora=False):
    """
    학습된 모델 로드

    Args:
        model_path: 학습된 모델 경로
        base_model: LoRA 사용 시 베이스 모델 경로
        is_lora: LoRA 어댑터 여부
    """
    if is_lora:
        # LoRA 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Full fine-tuned 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer


def generate_response(model, tokenizer, user_input, max_new_tokens=512):
    """
    사용자 입력에 대한 응답 생성
    """
    messages = [{"role": "user", "content": user_input}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    # 설정
    USE_LORA = True  # LoRA 모델 사용 여부

    if USE_LORA:
        model_path = "./outputs/qwen3-4b-mbti-lora"
        base_model = "Qwen/Qwen3-4B"
        model, tokenizer = load_model(model_path, base_model, is_lora=True)
    else:
        model_path = "./outputs/qwen3-4b-mbti-full"
        model, tokenizer = load_model(model_path, is_lora=False)

    print("모델 로드 완료!")

    # 대화 루프
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        response = generate_response(model, tokenizer, user_input)
        print(f"AI: {response}")
