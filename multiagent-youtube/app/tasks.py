from crewai import Task
from app.tools import get_youtube_tool


def get_tasks(youtube_url, agents):
    yt_tool = get_youtube_tool(youtube_url)

    research_task = Task(
        description="주제에 대한 비디오 내용을 식별합니다. 채널 비디오에서 비디오에 대한 자세한 정보를 얻습니다.",
        expected_output='비디오 콘텐츠를 기반으로 한 종합적인 3단락 보고서.',
        tools=[yt_tool],
        agent=agents['researcher']
    )

    write_task = Task(
        description="주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다.",
        expected_output='주제에 대한 유튜브 비디오의 정보를 요약하고 블로그 콘텐츠를 작성합니다.',
        tools=[yt_tool],
        agent=agents['writer']
    )

    translate_task = Task(
        description="주제에 대해 연구원이 작성해준 보고서를 기반으로 2단락 5000자로 요약하여서 한국어 콘텐츠를 작성합니다.",
        expected_output="주제에 대해 연구원이 작성해준 보고서를 기반으로 한국어 콘텐츠를 작성합니다.",
        agent=agents['translator']
    )

    return [research_task, write_task, translate_task]
