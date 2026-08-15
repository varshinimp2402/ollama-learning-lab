"""Minimal example using the official Ollama Python client."""

import os

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def main() -> None:
    client = ollama.Client(host=OLLAMA_HOST)

    try:
        response = client.chat(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Why is the sky blue?"},
            ],
        )
    except Exception as exc:
        raise SystemExit(
            f"Ollama request failed. Verify that {OLLAMA_HOST} is reachable "
            f"and that model '{MODEL}' is installed.\nDetails: {exc}"
        ) from exc

    print(response["message"]["content"])


if __name__ == "__main__":
    main()
