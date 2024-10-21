from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
import logging
from crew import create_crew

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 직렬화 함수 정의
def serialize_object(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return obj
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)
    
#JSON 데이터를 직렬화하여 문자열로 변환하는 부분
def custom_json_dumps(data):
    return json.dumps(data, default=serialize_object, ensure_ascii=False, indent=4)


# FastAPI 앱 생성
app = FastAPI(
    title="CrewAI Content Generation API",
    version="1.0",
    description="토픽을 기반으로 CrewAI를 사용하여 콘텐츠를 생성하는 API입니다.",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드를 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더를 허용
)


# 입력 데이터 모델 정의
class TopicInput(BaseModel):
    topic: str

# 콘텐츠 생성 엔드포인트
@app.post("/crewai")
async def crewai_endpoint(input: TopicInput):
    try:
        crew = create_crew() # Crew 생성
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff, {"topic": input.topic})
        serialized_result = custom_json_dumps(result)
        return JSONResponse(content=json.loads(serialized_result))
    except Exception as e:
        logger.error(f"CrewAI endpoint 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 앱 실행 (uvicorn 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)