"""
Quick connection test — run this first to verify the model is reachable.
    python test_connection.py
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL   = os.environ["DEEPSEEK_MODEL_NAME"]
API_URL = os.environ["DEEPSEEK_BASE_URL"] + "/v1/chat/completions"
API_KEY = os.environ["DEEPSEEK_API_KEY"]

# Uncomment if your work laptop needs a corporate proxy:
# PROXY_URL = os.environ["PROXY_URL"]
# PROXIES = {"http": PROXY_URL, "https": PROXY_URL}
PROXIES = None

print(f"Model:   {MODEL}")
print(f"API URL: {API_URL}")
print("Sending test message...")

try:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 50,
            "temperature": 0.0,
        },
        proxies=PROXIES,
        timeout=30,
    )
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"]
    print(f"\n✓ Success! Model replied: {reply}")

except requests.exceptions.ConnectionError:
    print("\n✗ Connection error — you may need to set PROXY_URL in .env")
except requests.exceptions.HTTPError as e:
    print(f"\n✗ HTTP error {response.status_code}: {response.text}")
except KeyError:
    print(f"\n✗ Unexpected response format: {response.json()}")
except Exception as e:
    print(f"\n✗ Error: {e}")
