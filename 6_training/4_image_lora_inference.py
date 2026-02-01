"""
Z-Image LoRA 추론 스크립트
학습된 LoRA 어댑터를 적용하여 이미지 생성
"""

import torch
from diffusers import ZImagePipeline
from peft import PeftModel


# 설정
MODEL_ID = "Tongyi-MAI/Z-Image"
LORA_PATH = "./outputs/z-image-pixel-lora"

# 픽셀 아트 스타일 고정 프롬프트
STYLE_SUFFIX = ", large, clearly visible pixels, chunky pixel blocks, low resolution look, limited color palette, no smooth gradients, no anti-aliasing, no blur, sharp pixel edges, retro 16-bit game style"


def load_pipeline(lora_path=None):
    """파이프라인 로드 (LoRA 옵션)"""
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        print(f"LoRA 로드 완료: {lora_path}")

    pipe.to("cuda")
    return pipe


def generate_image(pipe, prompt, negative_prompt="", save_path="output.png"):
    """이미지 생성"""
    full_prompt = prompt + STYLE_SUFFIX

    image = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
    ).images[0]

    image.save(save_path)
    print(f"이미지 저장: {save_path}")
    return image


if __name__ == "__main__":
    pipe = load_pipeline(LORA_PATH)

    prompt = "A cute cat sitting on a windowsill"
    negative_prompt = "blurry, realistic, photo"

    image = generate_image(pipe, prompt, negative_prompt, "pixel_cat.png")
    image.show()
