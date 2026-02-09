"""
Gradio 웹 UI - 고민 상담 & 위로 이미지 생성

실행 방법: python 2_gradio_app.py
※ 1_serving.py 서버가 먼저 실행되어 있어야 합니다
"""

import io
import requests
from PIL import Image
import gradio as gr


# FastAPI 서버 주소
API_URL = "http://localhost:8000"


# ============================================================
# 1) 화면 구성 (레이아웃)
# ============================================================

# Blocks: 자유롭게 화면을 구성할 수 있는 Gradio의 기본 틀
with gr.Blocks() as demo:

    # 제목과 설명
    gr.Markdown("# 🌿 고민 상담 & 힐링 이미지 생성기 🎨")
    gr.Markdown("고민을 입력하면 공감 응답을 드리고, 위로가 되는 픽셀아트 이미지를 만들어 드립니다 ✨")
    gr.Markdown(
        "### 📌 사용 방법\n"
        "1. 먼저 고민을 입력해주세요! 💬\n"
        "2. 고민을 다 입력했으면 **\"고민 입력 완료\"** 버튼을 눌러주세요! ✅\n"
        "3. 상담 응답 내용을 확인하면 힐링을 받을 수 있는 이미지 생성을 위해 **\"힐링 이미지 생성하기\"** 버튼을 눌러주세요! 🖼️"
    )

    # 이미지 프롬프트를 임시로 저장해두는 공간 (화면에는 보이지 않음)
    image_prompt_state = gr.State(value="")

    # Row: 가로로 나란히 배치 / Column: 세로로 쌓아서 배치
    with gr.Row():

        # --- 왼쪽 칼럼: 고민 입력 & 응답 ---
        with gr.Column():
            input_text = gr.Textbox(
                label="💬 고민을 입력해주세요",
                placeholder="여기에 고민을 적어주세요...",
                lines=3,
            )
            submit_btn = gr.Button("✅ 고민 입력 완료")
            reply_text = gr.Textbox(label="💌 상담 응답", interactive=False, lines=5)

        # --- 오른쪽 칼럼: 이미지 생성 ---
        with gr.Column():
            image_btn = gr.Button("🖼️ 힐링 이미지 생성하기")
            output_image = gr.Image(label="🎨 힐링 이미지", type="pil")


    # ============================================================
    # 2) API 연결 (버튼 클릭 시 동작)
    # ============================================================

    # 텍스트 API 호출 함수
    def get_reply(message):
        response = requests.post(f"{API_URL}/text", json={"message": message})
        data = response.json()
        # 공감 응답과 이미지 프롬프트를 각각 반환
        return data["reply"], data["image_prompt"]

    # 이미지 API 호출 함수
    def get_image(image_prompt):
        response = requests.post(f"{API_URL}/image", json={"prompt": image_prompt})
        image = Image.open(io.BytesIO(response.content))
        return image

    # "고민 입력 완료" 버튼 → 텍스트 API 호출
    submit_btn.click(
        fn=get_reply,
        inputs=[input_text],
        outputs=[reply_text, image_prompt_state],
    )

    # "힐링 이미지 생성하기" 버튼 → 이미지 API 호출
    image_btn.click(
        fn=get_image,
        inputs=[image_prompt_state],
        outputs=[output_image],
    )


# 서버 실행
demo.launch(server_name="0.0.0.0", server_port=9000)
