# tasks.py
from crewai import Task
from agents import planner, writer, editor, translator

# 기획 작업 정의
plan = Task(
    description=(
        "1. {topic}에 대한 최신 동향, 주요 인물, 주목할 만한 뉴스를 우선순위에 둡니다.\n"
        "2. 대상 청중을 식별하고 그들의 관심사와 어려움을 고려합니다.\n"
        "3. 소개, 주요 포인트, 행동 촉구를 포함한 자세한 콘텐츠 개요를 개발합니다.\n"
        "4. SEO 키워드와 관련 데이터 또는 소스를 포함합니다."
    ),
    expected_output=(
        "개요, 청중 분석, SEO 키워드, 리소스를 포함한 포괄적인 콘텐츠 계획 문서."
    ),
    agent=planner,
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

# 번역 작업 정의
translate_task = Task(
    description=(
        "주제에 대해 작성된 블로그 게시물을 2단락 5000자로 요약하여 한국어 콘텐츠를 작성합니다."
    ),
    expected_output="주제에 대해 작성된 블로그 게시물을 기반으로 한국어 콘텐츠를 작성합니다.",
    agent=translator,
    async_execution=False
)