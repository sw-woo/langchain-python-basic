from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI


# Ollama 사용 시 적용하는 방법
llm = ChatOpenAI(
    model="llama3.1:8b",                 # 사용할 언어 모델 지정
    base_url="http://localhost:11434/v1",  # 언어 모델의 베이스 URL
    api_key="NA"                         # API 키 (필요 없는 경우 'NA'로 설정)
)

# 콘텐츠 기획자 에이전트 정의
planner = Agent(
    role="콘텐츠 기획자",  # 역할 설정
    goal="{topic}에 대한 흥미롭고 사실적인 콘텐츠를 기획합니다",  # 목표 설정
    backstory=(
        "당신은 {topic}에 대한 블로그 기사를 기획하고 있습니다."
        "청중이 무언가를 배우고 정보에 입각한 결정을 내릴 수 있도록 도와주는 정보를 수집합니다."
        "블로그 게시물의 일부가 되어야 하는 자세한 개요와 관련된 주제 및 하위 주제를 준비해야 합니다."
        "당신의 작업은 이 주제에 대한 기사를 작성하는 콘텐츠 작가의 기초가 됩니다."
    ),  # 배경 스토리 설정
    llm=llm,                     # 언어 모델 지정
    allow_delegation=False,      # 위임 허용 여부
    verbose=True                 # 상세한 로그 출력 여부
)

# 콘텐츠 작가 에이전트 정의
writer = Agent(
    role="콘텐츠 작가",
    goal="주제: {topic}에 대한 통찰력 있고 사실적인 의견 기사를 작성합니다",
    backstory=(
        "당신은 {topic}에 대한 새로운 의견 기사를 작성하고 있습니다."
        "당신의 글은 콘텐츠 기획자의 작업을 기반으로 하며, 콘텐츠 기획자는 개요와 관련된 맥락을 제공합니다."
        "콘텐츠 기획자가 제공한 개요의 주요 목표와 방향을 따릅니다."
        "또한 콘텐츠 기획자가 제공한 정보로 뒷받침되는 객관적이고 공정한 통찰력을 제공합니다."
        "의견 진술과 객관적 진술을 구분하여 의견 기사에서 인정합니다."
    ),
    allow_delegation=False,
    llm=llm,
    verbose=True
)

# 편집자 에이전트 정의
editor = Agent(
    role="편집자",
    goal="주어진 블로그 게시물을 블로그 글쓰기 스타일에 맞게 편집합니다.",
    backstory=(
        "당신은 콘텐츠 작가로부터 블로그 게시물을 받는 편집자입니다."
        "당신의 목표는 블로그 게시물이 저널리즘의 모범 사례를 따르고,"
        "의견이나 주장 시 균형 잡힌 관점을 제공하며,"
        "가능하다면 주요 논란이 되는 주제나 의견을 피하도록 검토하는 것입니다."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

#open source llm 사용시 영어로 출력되는 부분을 언어를 감지해서 한국어로 변환시키는 부분
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



# 작업 정의

# 기획 작업 정의
plan = Task(
    description=(
        "1. {topic}에 대한 최신 동향, 주요 인물, 주목할 만한 뉴스를 우선순위에 둡니다.\n"
        "2. 대상 청중을 식별하고 그들의 관심사와 어려움을 고려합니다.\n"
        "3. 소개, 주요 포인트, 행동 촉구를 포함한 자세한 콘텐츠 개요를 개발합니다.\n"
        "4. SEO 키워드와 관련 데이터 또는 소스를 포함합니다."
    ),  # 작업에 대한 상세 설명
    expected_output=(
        "개요, 청중 분석, SEO 키워드, 리소스를 포함한 포괄적인 콘텐츠 계획 문서."
    ),  # 기대 출력물
    agent=planner,  # 이 작업을 수행할 에이전트 지정
)

# 작성 작업 정의
write = Task(
    description=(
        "1. 콘텐츠 계획을 사용하여 {topic}에 대한 매력적인 블로그 게시물을 작성합니다.\n"
        "2. SEO 키워드를 자연스럽게 통합합니다.\n"
        "3. 섹션/부제목은 매력적인 방식으로 적절하게 명명됩니다.\n"
        "4. 매력적인 소개, 통찰력 있는 본문, 요약 결론으로 구조화되었는지 확인합니다.\n"
        "5. 문법 오류와 브랜드의 음성에 맞게 교정합니다.\n"
    ),
    expected_output=(
        "마크다운 형식의 잘 작성된 블로그 게시물로, 각 섹션은 2~3개의 단락으로 구성되어 있으며, 출판 준비가 되어 있습니다."
    ),
    agent=writer,
)

# 편집 작업 정의
edit = Task(
    description=(
        "주어진 블로그 게시물을 문법 오류와 브랜드의 음성에 맞게 교정합니다."
    ),
    expected_output=(
        "마크다운 형식의 잘 작성된 블로그 게시물로, 각 섹션은 2~3개의 단락으로 구성되어 있으며, 출판 준비가 되어 있습니다."
    ),
    agent=editor
)

# 한국어로 변환하는 작업
translate_task = Task(
    description=(
        "주제에 대해 연구원이 작성해준 보고서를 기반으로 2단락 5000자로 요약하여서 한국어 콘텐츠를 작성합니다."),
    expected_output="주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다.",
    agent=translate_writter,
    async_execution=False,
    output_file= "translated-blog.md"
)

# Crew 생성 및 설정
crew = Crew(
    agents=[planner, writer, editor, translate_writter],  # 에이전트 목록
    tasks=[plan, write, edit, translate_task],         # 작업 목록
    verbose=True                       # 상세한 로그 출력 여부
)

# 입력값 설정
inputs = {
    "topic": "다중 에이전트 시스템 구축을 위한 LangGraph, Autogen 및 Crewai의 비교 연구"
}

# 작업 실행
result = crew.kickoff(inputs=inputs)

# 결과 출력
print(result)