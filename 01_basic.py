import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# .env 파일에서 API 키 읽어오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("환경 변수에서 OPENAI_API_KEY를 찾을 수 없습니다.")

# OpenAI 채팅 모델 생성
chat = ChatOpenAI(
    model="gpt-3.5-turbo", #or gpt-4 oepn-ai 모델 선택부분은 공식 홈페이지에서 확인 가능합니다. 현재는 gpt-3.5-turbo 모델을 사용하겠습니다.
    temperature=0, openai_api_key=OPENAI_API_KEY
)

# 질문 정의
question = "기분이 우울한 날에는 어떤것을 하면 기분이 나아질수 있을까?"

# 시스템에게 역할을 지정하고 메세지를 나의 메세지를 던지면 좀더 원하는 내용의 답변을 받을수 있습니다.
messages = [
    SystemMessage(
        content="너는 심리 상담가 입니다."
    ),
    HumanMessage(
        content= question
    ),
]

# 질문에 대한 응답 생성
try:
    response = chat.invoke(messages)
    print(response)
except Exception as e:
    print(f"Error: {e}")