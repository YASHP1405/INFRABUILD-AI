import os
import requests

keys = [os.environ.get("GROQ_API_KEY") # env variable
]

for k in keys:
    if not k:
        continue
    headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
    data = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hello"}]}
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    print(f"Key: {k[:10]}... Status: {r.status_code}")
    if r.status_code != 200:
        print(r.json())
