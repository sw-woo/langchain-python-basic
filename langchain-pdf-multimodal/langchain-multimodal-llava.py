import requests
import base64
import json


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_vision(prompt, image_path):
    base64_image = encode_image(image_path)

    payload = {
        "model": "llava:7b",
        "prompt": prompt,
        "images": [base64_image]
    }

    response = requests.post(
        "http://localhost:11434/api/generate", json=payload)

    print("Response: ", response)
    print("Status Code: ", response.status_code)
    print("Headers: ", response.headers)
    print("Content: ", response.content)

    if response.status_code == 200:
        try:
            # 응답을 줄 단위로 처리
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_line = json.loads(line)
                    if 'response' in json_line:
                        full_response += json_line['response']
            return full_response, response
        except json.JSONDecodeError as e:
            return f"JSON 파싱 오류: {str(e)}\n응답 내용: {response.text}"
    else:
        return f"Error: {response.status_code}, {response.text}"


# 사용 예시
image_path = "/Users/usermackbookpro/langchain-python/figures/figure-1-1.jpg"
prompt = "이 이미지에 대해 한국어로 설명해주세요."

result, response = generate_vision(prompt, image_path)
print(result)
print(response)
