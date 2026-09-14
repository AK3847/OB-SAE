"""Print the account's real rate limits, straight from the API response headers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import load_dotenv  # noqa: E402

import openai  # noqa: E402

src = load_dotenv()
print(f"env from {src}")

client = openai.OpenAI()
raw = client.chat.completions.with_raw_response.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=1,
)
for k, v in raw.headers.items():
    if "ratelimit" in k.lower():
        print(f"{k:40s} {v}")
