# 필요한 모듈을 가져옵니다.
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

# dotenv 모듈을 사용하여 .env 파일에서 환경 변수를 로드합니다.
from dotenv import load_dotenv
import os

# .env 파일에서 API 키를 읽어옵니다.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션을 생성합니다.
app = FastAPI(
    title="Langchain Server",  # 애플리케이션의 제목을 설정합니다.
    version="0.1",             # 애플리케이션의 버전을 설정합니다.
    description="simple langchain API Server"  # 애플리케이션의 설명을 설정합니다.
)

# OpenAI 모델을 설정합니다.
# ChatOpenAI 클래스를 사용하여 OpenAI 모델과의 상호작용을 설정합니다.
# 여기서는 API 키, 온도 설정, 모델 이름을 파라미터로 제공합니다.
model = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # OpenAI API 키를 사용하여 인증합니다.
    temperature=0.7,         # 응답의 창의성(무작위성) 수준을 설정합니다. 0.7은 중간 정도의 창의성을 의미합니다.
    model='gpt-3.5-turbo'    # 사용할 OpenAI 모델의 이름을 지정합니다.
)

# Langchain API 경로를 추가합니다.
# add_routes 함수를 사용하여 FastAPI 애플리케이션에 경로를 추가합니다.
add_routes(
    app,
    model,  # ChatOpenAI 클래스의 인스턴스를 사용하여 OpenAI 모델과 상호작용합니다.
    path="/openai"  # 이 경로로 요청이 들어오면 ChatOpenAI 인스턴스를 통해 처리됩니다.
)


# 소설 작성용 프롬프트 템플릿을 생성합니다.
prompt1 = ChatPromptTemplate.from_template("주제에 맞는 소설을 작성해줘 {topic}")

# 시 작성용 프롬프트 템플릿을 생성합니다.
prompt2 = ChatPromptTemplate.from_template("주제에 맞는 시를 작성해줘 {topic}")

# 소설 작성 API 경로를 추가합니다.
add_routes(
    app,
    prompt1 | model,  # 프롬프트 템플릿과 모델을 결합하여 요청을 처리합니다.
    path="/essay"  # 이 경로로 요청이 들어오면 소설을 작성합니다.
)

# 시 작성 API 경로를 추가합니다.
add_routes(
    app,
    prompt2 | model,  # 프롬프트 템플릿과 모델을 결합하여 요청을 처리합니다.
    path="/poem"  # 이 경로로 요청이 들어오면 시를 작성합니다.
)

# 애플리케이션을 실행합니다.
# 실행명령어 : python app.py
if __name__ == "__main__":
    # localhost:8000에서 애플리케이션을 실행합니다.
    uvicorn.run(app, host="localhost", port=8000)
