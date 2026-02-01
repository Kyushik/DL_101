"""
Z-Image LoRA 추론 스크립트 (DiffSynth Studio)
학습된 LoRA 어댑터를 적용하여 이미지 생성

설치: pip install diffsynth
"""

import torch
from diffsynth import ModelManager, ZImagePipeline


# 설정
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
LORA_PATH = "./outputs/z-image-pixel-lora/lora.safetensors"

# 픽셀 아트 스타일 고정 프롬프트
STYLE_SUFFIX = ", large, clearly visible pixels, chunky pixel blocks, low resolution look, limited color palette, no smooth gradients, no anti-aliasing, no blur, sharp pixel edges, retro 16-bit game style"


def load_pipeline(lora_path=None):
    """파이프라인 로드 (LoRA 옵션)"""
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")
    model_manager.load_models_from_huggingface(MODEL_ID)

    if lora_path:
        model_manager.load_lora(lora_path, lora_alpha=1.0)
        print(f"LoRA 로드 완료: {lora_path}")

    pipe = ZImagePipeline.from_model_manager(model_manager)
    return pipe


def generate_image(pipe, prompt, negative_prompt="", save_path="output.png"):
    """이미지 생성"""
    full_prompt = prompt + STYLE_SUFFIX

    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=8,  # Z-Image-Turbo는 8 스텝 권장
        cfg_scale=1.0,  # Turbo 모델은 cfg_scale=1 권장
    )

    image.save(save_path)
    print(f"이미지 저장: {save_path}")
    return image


if __name__ == "__main__":
    pipe = load_pipeline(LORA_PATH)

    prompt = input("만들고 싶은 이미지 내용을 영어로 입력하세요: ")
    negative_prompt = "blurry, realistic, photo"

    image = generate_image(pipe, prompt, negative_prompt, "prompt.png")
