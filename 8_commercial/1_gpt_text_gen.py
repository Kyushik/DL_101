import openai
from dotenv import load_dotenv

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

# GPT에 응답 요청 
response = openai.chat.completions.create(
    model="gpt-5.2",
    messages=[
        {"role": "system", "content": "당신은 사용자의 질문에 대해 유용하고 정확한 답변을 제공하는 AI입니다."},
        {"role": "user", "content": "서울 하루 여행 일정을 짜줘!"},
    ],
)

# 결과 출력
print(response.choices[0].message.content)