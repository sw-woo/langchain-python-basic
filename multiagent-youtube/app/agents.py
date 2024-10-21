from crewai import Agent
from app.tools import get_youtube_tool
from langchain_openai import ChatOpenAI

# gpt-4o-mini 모델 사용시 적용하는 방법 성능이 오픈소스 llama3.1:8b 모델 보다 좋습니다. 단점은 비용이 듭니다.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)


# LLM 설정
# llm = ChatOpenAI(
#     model="llama3.1:8b",
#     base_url="http://localhost:11434/v1",
#     api_key="NA"
# )

# 에이전트 생성 함수


def create_agents(youtube_url):
    yt_tool = get_youtube_tool(youtube_url)

    blog_researcher = Agent(
        role='유튜브 비디오 주제에 대한 블로그 연구원',
        goal='제공된 유튜브 동영상에서 주제를 파악하고 주제에 관한 관련 비디오를 분석합니다.',
        verbose=True,
        backstory="유튜브 콘텐츠 전문가로서 최대 2000자 이내로 한국어로 내용을 요약해서 제안을 제공합니다",
        tools=[yt_tool],
        allow_delegation=True,
        llm=llm
    )

    blog_writer = Agent(
        role='블로그 작가',
        goal='유튜브 비디오에 대한 매력적인 이야기를 서술합니다',
        verbose=True,
        backstory="유튜브 콘텐츠 전문가준 자료를 블로그 전문 작가로서 한국어로 내용을 요약해서 제안을 제공합니다",
        tools=[yt_tool],
        allow_delegation=False,
        llm=llm
    )

    translate_writer = Agent(
        role="translator",
        goal="Translate to korean",
        verbose=True,
        backstory="언어를 감지해서 한국어로 바꾸어서 작성해줘",
        allow_delegation=False,
        llm=llm
    )

    return {
        "researcher": blog_researcher,
        "writer": blog_writer,
        "translator": translate_writer
    }
