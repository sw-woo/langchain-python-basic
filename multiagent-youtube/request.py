import requests

url = 'http://localhost:8100/crewai'
data = {'youtube_url': 'https://www.youtube.com/watch?v=oSjMFyoLzNs'}
response = requests.post(url, json=data)
print(response.json())
