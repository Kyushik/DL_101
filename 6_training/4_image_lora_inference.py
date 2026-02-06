"""
Z-Image LoRA 추론 스크립트 (DiffSynth Studio)
학습된 LoRA 어댑터를 적용하여 이미지 생성

설치: pip install diffsynth
"""

import os
import torch
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig


# 설정 (로컬 모델 경로 사용)
MODEL_PATH = "./models/Tongyi-MAI/Z-Image-Turbo"
LORA_PATH = "./outputs/z-image-pixel-lora/epoch-1.safetensors"

# 픽셀 아트 스타일 고정 프롬프트
STYLE_SUFFIX = ", pxlstl"

def resolve_lora_path(path):
    """로컬 경로면 그대로, HuggingFace repo_id면 다운로드"""
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=path, filename="z-image-turbo-pixel-art-lora.safetensors")

def load_pipeline(lora_path=None):
    """파이프라인 로드 (LoRA 옵션)"""
    import glob

    # 로컬 파일 경로 직접 지정
    transformer_files = sorted(glob.glob(f"{MODEL_PATH}/transformer/*.safetensors"))
    text_encoder_files = sorted(glob.glob(f"{MODEL_PATH}/text_encoder/*.safetensors"))
    vae_files = sorted(glob.glob(f"{MODEL_PATH}/vae/*.safetensors"))

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=transformer_files),
            ModelConfig(path=text_encoder_files),
            ModelConfig(path=vae_files),
        ],
        tokenizer_config=ModelConfig(path=f"{MODEL_PATH}/tokenizer"),
    )

    if lora_path:
        lora_path = resolve_lora_path(lora_path)
        pipe.load_lora(pipe.dit, lora_path)
        print(f"LoRA 로드 완료: {lora_path}")

    return pipe


def generate_image(pipe, prompt, save_path="output.png"):
    """이미지 생성"""
    image = pipe(prompt=prompt + STYLE_SUFFIX, seed=42, rand_device="cuda")

    image.save(save_path)
    print(f"이미지 저장: {save_path}")
    return image


if __name__ == "__main__":
    pipe = load_pipeline(LORA_PATH)

    print("프롬프트를 입력하세요. 종료하려면 'quit' 또는 'q'를 입력하세요.")
    count = 0
    while True:
        prompt = input("\n프롬프트: ").strip()
        if prompt.lower() in ("quit", "q", "exit"):
            print("종료합니다.")
            break
        if not prompt:
            continue
        image = generate_image(pipe, prompt, f"{prompt}.png")
