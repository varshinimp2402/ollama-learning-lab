# Ollama Learning Lab

A small, Git-friendly collection of local LLM experiments built while learning **Ollama**, including direct API calls, the Ollama Python client, prompt-based categorization, a custom `Modelfile`, and a PDF RAG application with Streamlit + LangChain.

Everything is designed to run locally against an Ollama server.

## What is included

| Path | Purpose |
| --- | --- |
| `examples/http_generate.py` | Call Ollama's streaming HTTP API with `requests` |
| `examples/python_chat.py` | Chat using the official `ollama` Python package |
| `examples/grocery_categorizer.py` | Categorize a text file with a local model |
| `apps/pdf_rag_app.py` | Upload a PDF and query it using local RAG |
| `Modelfile` | Example custom Ollama model configuration |
| `data/grocery_list.txt` | Sample input for the categorizer |

## Prerequisites

- Python 3.10+
- Ollama installed and running
- Enough local RAM/VRAM for the model you choose

Verify Ollama is available:

```bash
ollama --version
```

Pull the models used by the examples:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

You can substitute another installed Ollama chat model by setting `OLLAMA_MODEL`.

## Setup

Clone the repository and enter it:

```bash
git clone <your-repository-url>
cd ollama-learning-lab
```

Create a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional: copy `.env.example` values into your shell environment if your Ollama host or model names differ from the defaults.

## Run the examples

Direct HTTP API:

```bash
python examples/01_http_generate.py
```

Ollama Python client:

```bash
python examples/02_python_chat.py
```

Grocery categorizer:

```bash
python examples/03_grocery_categorizer.py
```

The categorized result is written to `outputs/categorized_grocery_list.txt`. Generated outputs are intentionally ignored by Git.

You can also provide your own files:

```bash
python examples/03_grocery_categorizer.py --input path/to/items.txt --output outputs/result.txt
```

## Run the PDF RAG app

Make sure both models are available first:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Start Streamlit:

```bash
streamlit run apps/pdf_rag_app.py
```

Open the local Streamlit URL, upload a PDF, and ask questions about it. PDFs are not committed to the repository by default because they may contain private or copyrighted material.

## Build the custom model

The included `Modelfile` creates a simple assistant persona based on `llama3.2`:

```bash
ollama create anna-assistant -f Modelfile
ollama run anna-assistant
```

To use it in the Python examples, set the model environment variable.

### Windows PowerShell

```powershell
$env:OLLAMA_MODEL="anna-assistant"
```

### macOS / Linux

```bash
export OLLAMA_MODEL="anna-assistant"
```

## Configuration

The project recognizes these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Chat/generation model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG |

For a remote Ollama server on your private network, set `OLLAMA_HOST` to that machine's reachable IP/hostname and port.

## Repository hygiene

This repository intentionally excludes:

- `.venv/` and other local environments
- macOS/Windows metadata
- generated model outputs
- PDFs placed under `data/`
- Chroma/local database files
- `.env` files and Streamlit secrets

No API keys are required for the default local Ollama workflow.

## Ideas for extending the project

- Add a chat history to the RAG app
- Add source/page citations to RAG answers
- Compare different local embedding models
- Add automated tests and linting
- Expose the RAG workflow as a FastAPI endpoint for local coding agents
- Point `OLLAMA_HOST` at a LAN-hosted Ollama server
