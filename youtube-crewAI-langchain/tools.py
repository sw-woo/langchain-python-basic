from crewai_tools import YoutubeVideoSearchTool

# Initialize the tool with a specific Youtube channel handle to target your search
# yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')
yt_tool = YoutubeVideoSearchTool(
    youtube_video_url='https://youtu.be/Y-JY4LC4S-4?si=54xWnViR0P--cxV0')
