import requests
import json

url = 'http://localhost:8100/crewai/stream'
data = {'youtube_url': 'https://www.youtube.com/watch?v=Z8l7C031xkM'}
response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        try:
            json_line = json.loads(decoded_line)
            print(json_line)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")
