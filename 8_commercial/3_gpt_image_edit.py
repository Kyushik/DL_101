import base64
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

client = OpenAI()

prompt = """이 이미지를 애니메이션 캐릭터로 변경해줘"""

result = client.images.edit(
    model="gpt-image-1.5",
    image=[
        open("sample.png", "rb"),
    ],
    size="1024x1024",
    prompt=prompt
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("gpt_img_edit_output.png", "wb") as f:
    f.write(image_bytes)