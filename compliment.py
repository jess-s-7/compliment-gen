# ---------- Imports and metadata ----------
# compliment.py
# AI-Powered Compliment Generator
# Author: Jess Rhiannon
# Description: Generates a short compliment using OpenAI's API with random variety,
# retries/backoff on transient errors, and a local fallback if the API isn't available.

import os
import sys
import json
import urllib.request
import urllib.error
import random
import time

# ---------- Input (optional name) ----------
name = sys.argv[1] if len(sys.argv) > 1 else None

# ---------- Local fallback compliments ----------
FALLBACKS = [
    "You're a natural at turning complexity into clarity.",
    "You make tough problems look easy.",
    "Your curiosity makes everything you touch more interesting.",
    "You bring warmth to technical conversations.",
    "You're the kind of person who makes collaboration a joy.",
]

# ---------- Environment (API key & optional org) ----------
API_KEY = os.getenv("OPENAI_API_KEY")
ORG_ID = os.getenv("OPENAI_ORG")  # optional; set if your account requires it

# ---------- Randomization setup ----------
nonce = random.randint(100000, 999999)
temperature = round(random.uniform(0.9, 1.3), 2)
top_p = round(random.uniform(0.8, 1.0), 2)

twists = [
    "Make it sound different every time.",
    "Give it a whimsical tone.",
    "Add a touch of humor.",
    "Keep it poetic but short.",
    "Make it gentle and kind.",
]

# ---------- Request payload ----------
payload = {
    "model": "gpt-3.5-turbo",  # widely available; switch if your account supports another
    "messages": [
        {
            "role": "system",
            "content": "You are a kind and clever assistant who writes original, encouraging one-line compliments.",
        },
        {
            "role": "user",
            "content": (
                f"Write one short compliment for: {name or 'a reader'}. "
                f"{random.choice(twists)} "
                f"Each time you’re asked, vary the wording. Nonce: {nonce}"
            ),
        },
    ],
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": 50,
}

# ---------- The 'get_ai_compliment()' function ----------
def get_ai_compliment() -> str:
    """Return a compliment string. Uses OpenAI if available; otherwise local fallback."""
    if not API_KEY:
        print("[local – no API key]", end=" ")
        return random.choice(FALLBACKS)

    max_retries = 3
    backoff = 2  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            }
            if ORG_ID:
                headers["OpenAI-Organization"] = ORG_ID

            req = urllib.request.Request(
                url="https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"].strip()
                print("[AI]", end=" ")
                return text

        except urllib.error.HTTPError as e:
            # Read a short snippet of the error body (optional, helpful for debugging)
            snippet = ""
            try:
                snippet = e.read().decode(errors="ignore")[:100]
            except Exception:
                pass

            # Retry on common transient codes
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = backoff * attempt
                print(f"[retry {attempt} after {wait}s – HTTP {e.code}]", end=" ")
                time.sleep(wait)
                continue

            print(f"[fallback – HTTP {e.code}]", end=" ")
            return random.choice(FALLBACKS)

        except (urllib.error.URLError, TimeoutError):
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"[retry {attempt} after {wait}s – network]", end=" ")
                time.sleep(wait)
                continue
            print("[fallback – network]", end=" ")
            return random.choice(FALLBACKS)

        except Exception:
            print("[fallback – unknown]", end=" ")
            return random.choice(FALLBACKS)

    # Safety net — should not reach here
    print("[fallback – exhausted]", end=" ")
    return random.choice(FALLBACKS)

# ---------- Print the compliment ----------
if __name__ == "__main__":
    compliment = get_ai_compliment()
    print(compliment)
