"""Minimal example using Ollama's HTTP streaming API."""

import json
import os

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def main() -> None:
    payload = {
        "model": MODEL,
        "prompt": "Tell me a short story and make it funny.",
        "stream": True,
    }

    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            print("Generated text:\n")

            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                print(chunk.get("response", ""), end="", flush=True)

            print()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running and the model is available.\n"
            f"Details: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
