import os
from crewai import Agent
from tools import yt_tool
# from langchain_ollama.llms import OllamaLLM
# from langchain.llms import Ollama

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# groq api 서비스를 이용하여서 라마 사용하기 단점 무료는 하루 제한량이 정해져있음
#
# load_dotenv()
# llm = ChatGroq(
#     api_key=os.getenv("groq_api_key"),
#     model="llama-3.1-8b-instant"
# )


# ollama 사용시 적용하는 방법
llm = ChatOpenAI(
    model="llama3.1:8b",
    base_url="http://localhost:11434/v1",
    api_key="NA"
)

# gpt4 모델이 기본으로 설정되어서 작동 됩니다.
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"


# 시니어 블로그 콘텐츠 연구원 생성

blog_researcher = Agent(
    role='유튜브 비디오 주제에 대한 블로그 연구원',
    goal='제공된 유튜브 동영상에서 주제를 파악하고 주제에 관한 관련 비디오를 분석 합니다.',
    verbose=True,
    memory=True,
    backstory=(
        "유튜브 콘텐츠 전문가로서 최대 3000자 이내로 한국어로 내용을 요약해서 제안을 제공합니다"
        "모든 콘텐츠는 한국어로 작성합니다."
    ),
    tools=[yt_tool],
    allow_delegation=True,
    llm=llm
)

# 유튜브 도구를 사용한 시니어 블로그 작가 에이전트 생성

blog_writer = Agent(
    role='블로그 작가',
    goal='유튜브 비디오에 대한 매력적인 이야기를 서술합니다',
    verbose=True,
    memory=True,
    backstory=(
        "유튜브 콘텐츠 전문가준 자료를 블로그 전문 작가로서 한국어로 내용을 요약해서 제안을 제공합니다"
        # "언어를 감지해서 한국어로 바꾸어서 제공합니다. 무조건"
        # "복잡한 주제를 단순화하는 재능을 가지고 있으며, 흥미롭고 교육적인 내용으로 독자를 사로잡는"
        "이야기를 만들어 내면서 새로운 발견을 이해하기 쉬운 방식으로 조명합니다."
        # "최대 1000자 이내로 요약해서 정리해서 작성해줘!"
    ),
    tools=[yt_tool],
    allow_delegation=False,
    llm=llm
)

translate_writter = Agent(
    role="translator",
    goal="Translate to korean",
    verbose=True,
    memory=True,
    backstory=(
        "언어를 감지해서 한국어로 바꾸어서 작생해줘"
    ),
    allow_delegation=False,
    llm=llm
)
