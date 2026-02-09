"""
고민 상담 & 위로 이미지 생성 웹 서비스

실행 방법: uvicorn 1_serving:app
테스트 방법: 브라우저에서 http://localhost:8000/docs 접속
"""

import os
import io
import torch

os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download

app = FastAPI()


# ============================================================
# 1) 텍스트 모델 준비 (Qwen3-4B + 공감 LoRA)
# ============================================================

BASE_MODEL = "Qwen/Qwen3-4B"
LORA_MODEL = "mks0813/qwen3-4b-mbti-f-style-lora"

# 기본 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# 공감 스타일 LoRA 붙이기
model = PeftModel.from_pretrained(model, LORA_MODEL)


# ============================================================
# 2) 이미지 모델 준비 (Z-Image-Turbo + 픽셀아트 LoRA)
# ============================================================

IMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
IMAGE_LORA = "mks0813/z-image-turbo-pixel-art-lora"
STYLE_SUFFIX = ", pxlstl"  # 픽셀아트 스타일을 적용하는 키워드

# Z-Image 파이프라인 불러오기 (HuggingFace에서 자동 다운로드)
image_pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=IMAGE_MODEL, origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id=IMAGE_MODEL, origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id=IMAGE_MODEL, origin_file_pattern="vae/*.safetensors"),
    ],
)

# 픽셀아트 LoRA 붙이기
lora_file = hf_hub_download(repo_id=IMAGE_LORA, filename="epoch-1.safetensors")
image_pipe.load_lora(image_pipe.dit, lora_file)


# ============================================================
# 3) API 입력/출력 형태 정의
# ============================================================

class TextRequest(BaseModel):
    message: str  # 사용자의 고민

class TextResponse(BaseModel):
    reply: str         # 공감 응답
    image_prompt: str  # 이미지 생성용 프롬프트

class ImageRequest(BaseModel):
    prompt: str  # 이미지 생성용 프롬프트


# ============================================================
# 4) 텍스트 생성 함수
# ============================================================

def generate_text(messages, max_new_tokens=512):
    """메시지를 받아서 텍스트를 생성하는 함수"""
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
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 부분을 잘라내고 생성된 부분만 텍스트로 변환
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


# ============================================================
# 5) API 엔드포인트
# ============================================================

@app.post("/text", response_model=TextResponse)
def text_api(req: TextRequest):
    """고민을 보내면 공감 응답 + 이미지 프롬프트를 돌려줍니다"""

    # (1) LoRA를 켜고 → 공감 스타일로 응답 생성
    model.enable_adapter_layers()
    reply = generate_text([{"role": "user", "content": req.message}])

    # (2) LoRA를 끄고 → 기본 모델로 이미지 프롬프트 생성
    model.disable_adapter_layers()
    prompt_request = (
        f"User's concern: {req.message}\n"
        f"Counselor's response: {reply}\n\n"
        "Based on the conversation above, write a short English image prompt "
        "to generate a cute, cheerful, and healing illustration. "
        "The image should feel bright, warm, and positive. "
        "Keep it brief like the examples below. Output only the prompt.\n\n"
        "Examples:\n"
        "- A smiling baby bear playing with fallen leaves in a sunny autumn forest\n"
        "- A happy hamster sleeping peacefully on a cozy blanket under warm light\n"
        "- A boy riding a bicycle over a green hill under a bright rainbow\n"
    )
    image_prompt = generate_text([{"role": "user", "content": prompt_request}])

    return TextResponse(reply=reply, image_prompt=image_prompt)


@app.post("/image")
def image_api(req: ImageRequest):
    """프롬프트를 보내면 픽셀아트 위로 이미지를 생성합니다"""

    # 프롬프트 뒤에 픽셀아트 스타일 키워드를 붙여서 이미지 생성
    image = image_pipe(
        prompt=req.prompt + STYLE_SUFFIX,
        rand_device="cuda",
    )

    # 생성된 이미지를 PNG로 변환해서 전송
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")
