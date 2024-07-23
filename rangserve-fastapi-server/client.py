import requests

# 실행명령어 : python client.py
response1 = requests.post("http://localhost:8000/essay/invoke",
                          json={'input': {'topic': "행복에 대해서"}})


response2 = requests.post("http://localhost:8000/poem/invoke",
                          json={'input': {'topic': "행복에 대해서"}})

print(response1.json())

print(response2.json())
