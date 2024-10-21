from crewai_tools import YoutubeVideoSearchTool

tool = YoutubeVideoSearchTool(
    youtube_video_url='https://youtu.be/-KTb3zivHXk?si=85ZVCGYshzrunpWm'
)

print(tool)

