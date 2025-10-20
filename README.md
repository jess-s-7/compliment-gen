# Build an AI-Powered Compliment Generator in Python

Learn how to use the OpenAI’s Chat Completions API to generate one-line compliments, handle rate limits, and provide reliable fallbacks.

**Note**: This tutorial is for macOS/Unix systems. 

> **Who this guide is for:**  
> Developers or technical writers with basic Python familiarity who want to explore AI text generation in a lightweight, transparent way — no frameworks or complex dependencies required.

> **Estimated time to complete:** 10–15 minutes

---

## Before you begin

Before you start, make sure you have the following:

- **Python 3.8+** installed ([download](https://www.python.org/downloads/)).
- **OpenAI API key** from your [dashboard](https://platform.openai.com/account/api-keys).
- **OpenAI Org ID** Optional; set if your account requires it
- **Terminal**

To verify your Python version:
```bash
python --version
```
You should see output similar to:
```
Python 3.11.0
```

---

## Set up your environment
```bash
# ---------- Environment setup ----------
mkdir ~/Desktop/compliment-gen
cd ~/Desktop/compliment-gen
touch compliment.py
# Set your API key for this terminal *session*:
export OPENAI_API_KEY=your-real-api-key
```

> Replace `your-real-api-key` with your actual key.
> The `export` command applies only to the current shell session.
> To make it persistent, add the same line to `~/.zshrc`, then `source` that file.


---

## Write your compliment generator

Let's build a one-line complement generator with offline-fallbacks using Python. 

---

### 1. Imports and metadata
- This tutorial only uses the Python standard library — no external dependencies required.
- `urllib` handles HTTP requests directly to the OpenAI API.
- `random` for randomization of each compliment.

```python
# ---------- Imports and metadata ----------
# compliment.py
# AI-Powered Compliment Generator
# Author: Jess Rhiannon
# Description: Generates a short, random compliment using OpenAI's API,
# retries/backoff on transient errors, and a local fallback if the API isn't available.

import os
import sys
import json
import urllib.request
import urllib.error
import random
import time
```
---

### 2. Command-line input

Allow the script to take an optional name argument (for example, `python compliment.py Jess`).

```python
# ---------- Input (optional name) ----------
name = sys.argv[1] if len(sys.argv) > 1 else None
```

If no name is provided, it defaults to `None`, and the compliment is addressed to “a reader.”

---

### 3. Local fallback compliments

These local strings ensure that the program always produces a compliment, even without internet access or an API key.

```python
# ---------- Local fallback compliments ----------
FALLBACKS = [
    "You're a natural at turning complexity into clarity.",
    "You make tough problems look easy.",
    "Your curiosity makes everything you touch more interesting.",
    "You bring warmth to technical conversations.",
    "You're the kind of person who makes collaboration a joy.",
]
```
---

### 4. Environment variables

Load your OpenAI credentials from environment variables. ORG_ID is optional; set if your account requires it.

```python
# ---------- Environment (API key & optional org) ----------
API_KEY = os.getenv("OPENAI_API_KEY")
ORG_ID = os.getenv("OPENAI_ORG")  # optional; set if your account requires it
```

- `OPENAI_API_KEY` is required to authenticate with the API.
- `OPENAI_ORG` is optional, used only if your organization requires it.

---

### 5. Randomization setup
This function adds randomness so each compliment feels fresh and natural, creating novelty and variety instead of repeating the same hardcoded messages.

```python
# ---------- Randomization setup ----------
# Introduce randomness for creative variety on each run.

# Prevents identical responses for the same prompt.
nonce = random.randint(100000, 999999) 

# Controls creatvity. Higher is more creativity.
temperature = round(random.uniform(0.9, 1.3), 2) 

# Controls predictability. Lower is more common compliments, higher is less predictable compliments.
top_p = round(random.uniform(0.8, 1.0), 2) 

# Prompt variations that alter tone and style.
twists = 
 [
    "Make it sound different every time.",
    "Give it a whimsical tone.",
    "Add a touch of humor.",
    "Keep it poetic but short.",
    "Make it gentle and kind.",
]
```



---

### 6. Building the API payload

Construct the request body for the OpenAI Chat Completions API.

```python
# ---------- API payload ----------
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
    # Limits the resopnse length. Each token is about a few characters or a part of a word.
    "max_tokens": 50,
}
```

- Uses the **Chat Completions API** (`/v1/chat/completions`)
- Includes both a **system** message (sets assistant tone) and a **user** message (defines the request).

---

### 7. The `get_ai_compliment()` function

This function manages retries, handles API/network errors, and provides fallback compliments. 

**What this does:**
- Retries transient API errors (`429`, `500–504`).
- Uses exponential backoff via `wait = backoff * attempt`.
- Falls back gracefully so the user always gets a compliment.
- Prints `[AI]`, `[local]`, or `[fallback]` prefixes to indicate response type.

```python
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
            snippet = ""
            try:
                snippet = e.read().decode(errors="ignore")[:100]
            except Exception:
                pass

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

    print("[fallback – exhausted]", end=" ")
    return random.choice(FALLBACKS)
```



---

### 8. Printing compliment

```python
# ---------- Print the compliment ----------
if __name__ == "__main__":
    compliment = get_ai_compliment()
    print(compliment)
```

## 9. Run the script 

```bash
# ---------- Run the script ----------
cd ~/Desktop/compliment-gen
python3 compliment.py
python3 compliment.py Jess  # Prints a personalized compliment

```


---

## Handling errors

The script prints a prefix so you can tell what happened:

| Prefix | Meaning |
|--------|----------|
| `[AI]` | Successful response from OpenAI |
| `[local]` | No API key set (offline/local fallback) |
| `[fallback]` | Error after retries (429/other failures) |

> If you see `[fallback]`, wait a few seconds and retry. For rate‑limit reliability, add billing or credits in your [OpenAI account](https://platform.openai.com/account/billing).

---

## Next steps

- Add a `--style` flag for *funny*, *wholesome*, or *poetic* tones.
- Create a simple web interface to display compliments visually.
