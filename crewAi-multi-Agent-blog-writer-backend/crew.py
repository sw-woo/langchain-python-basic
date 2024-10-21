from crewai import Crew, Process
from agents import planner, writer, editor, translator
from tasks import plan, write, edit, translate_task

def create_crew():
    # Crew 생성
    crew = Crew(
        agents=[planner, writer, editor, translator],  # 에이전트 리스트
        tasks=[plan, write, edit, translate_task],     # 수행할 작업 리스트
        process=Process.sequential,                    # 작업 순서: 순차적으로 진행
        verbose=True                                   # 작업 상세 로그 출력
    )
    return crew
