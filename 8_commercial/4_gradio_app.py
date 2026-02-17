"""
Gradio 웹 UI - 고민 상담 & 위로 이미지 생성 (OpenAI API 직접 호출 버전)

실행 방법: python 4_gradio_app.py
※ .env 파일에 OPENAI_API_KEY가 설정되어 있어야 합니다
"""

import json
import base64
import io

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import gradio as gr

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

client = OpenAI()


# ============================================================
# 1) 화면 구성 (레이아웃)
# ============================================================

with gr.Blocks() as demo:

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
    # 2) OpenAI API 연결 (버튼 클릭 시 동작)
    # ============================================================

    def get_reply(message):
        """GPT에게 고민 상담 요청 → JSON으로 공감 응답 + 이미지 프롬프트를 받아옴"""
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 사용자의 고민을 듣고 공감해주는 따뜻한 상담사야.\n"
                        "사용자의 고민에 대해 MBTI의 F 스타일로 공감하는 짧은 응답을 해줘.\n"
                        "적절한 이모지를 사용하면서 친근한 스타일의 응답을 해줘.\n"
                        "또한, 사용자의 고민을 위로할 수 있는 픽셀 스타일 이미지를 생성하기 위한 영어 프롬프트도 만들어줘.\n\n"
                        "반드시 아래와 같은 한 줄짜리 JSON 형식으로만 응답해:\n"
                        '{"reply": "공감 응답 내용", "image_prompt": "pixel art style, 이미지 설명 영어 프롬프트"}\n\n'
                        "절대 코드 블록(```)을 사용하지 마. 줄바꿈 없이 순수한 JSON 한 줄만 출력해."
                    ),
                },
                {"role": "user", "content": message},
            ],
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        print(f"공감 응답: {data['reply']}")
        print(f"이미지 프롬프트: {data['image_prompt']}")
        return data["reply"], data["image_prompt"]

    def get_image(image_prompt):
        """OpenAI 이미지 생성 API로 픽셀 스타일 이미지 생성"""
        result = client.images.generate(
            model="gpt-image-1.5",
            prompt=image_prompt,
            n=1,
            size="1024x1024",
        )

        image_bytes = base64.b64decode(result.data[0].b64_json)
        image = Image.open(io.BytesIO(image_bytes))
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
demo.launch(server_name="0.0.0.0", server_port=9090)
