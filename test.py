import requests
import json

url = "http://127.0.0.1:5000/generate"

payload = {
    "initial_text": "king:",
    "max_length": 200
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(payload), headers=headers, stream=True)

if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=1):
        if chunk:
            print(chunk.decode('utf-8'), end='')
else:
    print(f"Error: {response.status_code}")
