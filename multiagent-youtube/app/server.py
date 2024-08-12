from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio
import json
import logging
from crewai import Crew, Process
from .agents import create_agents
from .tasks import get_tasks

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serialize_object(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return obj
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def custom_json_dumps(data):
    return json.dumps(data, default=serialize_object, ensure_ascii=False, indent=4)


async def stream_result(crew):
    try:
        for result_chunk in crew.kickoff():
            json_chunk = json.dumps(
                {"chunk": result_chunk}, default=serialize_object)
            yield f"{json_chunk}\n"
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in streaming result: {str(e)}")
        yield json.dumps({"error": str(e)}) + "\n"


def create_crew(youtube_url):
    agents = create_agents(youtube_url)
    tasks = get_tasks(youtube_url, agents)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )


app = FastAPI(
    title="CrewAI LangServe API",
    version="1.0",
    description="A LangServe API for CrewAI-based content generation from YouTube videos",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YouTubeInput(BaseModel):
    youtube_url: str


@app.post("/crewai/stream")
async def crewai_stream_endpoint(input: YouTubeInput):
    try:
        crew = create_crew(input.youtube_url)
        return StreamingResponse(stream_result(crew), media_type="application/x-ndjson")
    except Exception as e:
        logger.error(f"Error in CrewAI chain: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crewai")
async def crewai_endpoint(input: YouTubeInput):
    try:
        crew = create_crew(input.youtube_url)
        result = crew.kickoff()
        serialized_result = custom_json_dumps(result)
        return JSONResponse(content=json.loads(serialized_result))
    except Exception as e:
        logger.error(f"Error in CrewAI chain: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
