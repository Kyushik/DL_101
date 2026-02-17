import base64
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

client = OpenAI()

img = client.images.generate(
    model="gpt-image-1.5",
    prompt="화산 위에 떠 있는 유리 테이블에서 고양이와 체스를 두는 네 개의 손을 가진 사람",
    n=1,
    size="1024x1024"
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("gpt_img_gen_output.png", "wb") as f:
    f.write(image_bytes)