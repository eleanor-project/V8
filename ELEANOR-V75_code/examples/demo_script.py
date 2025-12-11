import requests

response = requests.post(
    "http://localhost:8000/evaluate",
    json={"text": "Should we implement this facial recognition feature?"}
)
print(response.json())
