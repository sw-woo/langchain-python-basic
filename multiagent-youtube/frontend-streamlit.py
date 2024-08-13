import streamlit as st
import requests
import json

# Streamlit 설정
st.set_page_config(page_title="CrewAI 콘텐츠 생성기", layout="centered")
st.title("CrewAI 콘텐츠 생성기")
st.write("YouTube 비디오에서 콘텐츠를 생성하기 위해 CrewAI의 강력한 백엔드를 사용하세요.")

# YouTube URL 입력
youtube_url = st.text_input("YouTube URL 입력", "")

# 결과를 보여줄 부분
result_container = st.empty()
download_container = st.empty()


def format_output(result):
    """
    주어진 결과에서 'raw' 항목만 출력하고, 이를 텍스트 파일로 다운로드할 수 있도록 합니다.
    """
    if "raw" in result:
        # raw 항목 출력
        result_container.subheader("원본 텍스트 (Raw)")
        result_container.write(result["raw"])

        # raw 텍스트 형식으로 다운로드 버튼 추가
        raw_data = result["raw"]
        download_container.download_button(
            label="Download Raw Data as Text File",
            data=raw_data,
            file_name='raw_data.txt',
            mime='text/plain'
        )
    else:
        result_container.warning("결과에 'raw' 항목이 없습니다.")


def call_single_response_api(youtube_url):
    url = 'http://localhost:8100/crewai'
    data = {'youtube_url': youtube_url}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# 버튼 클릭 시 API 호출
if st.button("콘텐츠 생성"):
    if youtube_url:
        result_container.empty()  # 이전 결과 초기화
        download_container.empty()  # 다운로드 버튼 초기화
        with st.spinner("처리 중입니다... 잠시만 기다려 주세요."):
            result = call_single_response_api(youtube_url)
            if "error" in result:
                result_container.error(result["error"])
            else:
                format_output(result)
    else:
        st.warning("유효한 YouTube URL을 입력해 주세요.")
