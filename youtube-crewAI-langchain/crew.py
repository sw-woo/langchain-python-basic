from crewai import Crew, Process
from agents import blog_researcher, blog_writer, translate_writter
from tasks import research_task, write_task, translate_task


# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents=[blog_researcher, blog_writer, translate_writter],
    tasks=[research_task, write_task, translate_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    # memory=True,
    # cache=True,
    max_rpm=15,
    # share_crew=True,
    verbose=True
)

# start the task execution process with enhanced feedback
# result = crew.kickoff(inputs={'topic': '너가 알아서 영상 주제를 파악해줘'})
result = crew.kickoff()
print(result)
