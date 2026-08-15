"""Categorize a grocery list with a local Ollama model."""

import argparse
import os
from pathlib import Path

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "grocery_list.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "categorized_grocery_list.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_prompt(items: str) -> str:
    return f"""You are an assistant that categorizes and sorts grocery items.

Here is the grocery list:

{items}

Please:
1. Group the items into sensible categories such as Produce, Dairy, Pantry, Household, and Beverages.
2. Sort items alphabetically within each category.
3. Return only the organized list using clear Markdown headings and bullet points.
"""


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    items = args.input.read_text(encoding="utf-8").strip()
    if not items:
        raise SystemExit(f"Input file is empty: {args.input}")

    client = ollama.Client(host=OLLAMA_HOST)

    try:
        response = client.generate(model=MODEL, prompt=build_prompt(items))
    except Exception as exc:
        raise SystemExit(
            f"Ollama request failed. Verify that {OLLAMA_HOST} is reachable "
            f"and that model '{MODEL}' is installed.\nDetails: {exc}"
        ) from exc

    generated_text = response.get("response", "").strip()
    if not generated_text:
        raise SystemExit("Ollama returned an empty response.")

    print(generated_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated_text + "\n", encoding="utf-8")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
