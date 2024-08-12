import streamlit as st
import requests
import json

# Streamlit 설정
st.set_page_config(page_title="CrewAI 콘텐츠 생성기", layout="centered")
st.title("CrewAI 콘텐츠 생성기")
st.write("YouTube 비디오에서 콘텐츠를 생성하기 위해 CrewAI의 강력한 백엔드를 사용하세요.")

# YouTube URL 입력
youtube_url = st.text_input("YouTube URL 입력", "")

# API 선택
api_option = st.radio("응답 유형 선택", options=["단일 응답", "스트리밍 응답"])

# 결과를 보여줄 부분
result_container = st.empty()


def call_single_response_api(youtube_url):
    url = 'http://localhost:8100/crewai'
    data = {'youtube_url': youtube_url}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def call_streaming_response_api(youtube_url):
    url = 'http://localhost:8100/crewai/stream'
    data = {'youtube_url': youtube_url}
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        result = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                result.append(chunk['chunk'])
                result_container.text("\n".join(result))
        return result
    except requests.exceptions.RequestException as e:
        result_container.error(f"오류: {str(e)}")
        return {"error": str(e)}


# 버튼 클릭 시 API 호출
if st.button("콘텐츠 생성"):
    if youtube_url:
        result_container.empty()  # 이전 결과 초기화
        with st.spinner("처리 중입니다... 잠시만 기다려 주세요."):
            if api_option == "단일 응답":
                result = call_single_response_api(youtube_url)
                if "error" in result:
                    result_container.error(result["error"])
                else:
                    result_container.json(result)
            elif api_option == "스트리밍 응답":
                call_streaming_response_api(youtube_url)
    else:
        st.warning("유효한 YouTube URL을 입력해 주세요.")
