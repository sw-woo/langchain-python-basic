from crewai import Task
from tools2 import yt_tool
from agents2 import blog_researcher, blog_writer, translate_writter
import os
from crewai_tools import YoutubeVideoSearchTool

youtube_url = "https://youtu.be/RtJcPfuj_Wg?si=SUFTm2yARj6KESsO"

# 연구 과제
research_task = Task(
    description=(
        "주제에 대한 비디오 내용을 식별합니다."
        "채널 비디오에서 비디오에 대한 자세한 정보를 얻습니다."
    ),
    expected_output='비디오 콘텐츠의을(를) 기반으로 한 종합적인 3단락 보고서.',
    tools=[YoutubeVideoSearchTool(
        youtube_video_url=youtube_url)],
    agent=blog_researcher,
)

# 언어 모델 설정을 포함한 작성 과제

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
write_task = Task(
    description=(
        "주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다."
    ),
    expected_output='주제에 대한 유튜브 비디오의 정보를 요약하고 블로그 콘텐츠를 작성합니다.',
    tools=[YoutubeVideoSearchTool(
        youtube_video_url=youtube_url)],
    agent=blog_writer,
    async_execution=False,
    # 출력 커스터마이제이션 예시
    # output_file="new-blog-post.md"
    # output_file=f"{SCRIPT_DIR}/new-blog-post.md"
)

translate_task = Task(
    description=(
        "주제에 대해 연구원이 작성해준 보고서를 기반으로 2단락 5000자로 요약하여서 한국어 콘텐츠를 작성합니다."),
    expected_output="주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다.",
    agent=translate_writter,
    async_execution=False,
    output_file="translated-blog.md"
)
