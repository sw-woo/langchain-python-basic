from crewai_tools import YoutubeVideoSearchTool


def get_youtube_tool(youtube_url):
    return YoutubeVideoSearchTool(youtube_video_url=youtube_url)
