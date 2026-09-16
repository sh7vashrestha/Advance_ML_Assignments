# Agentic Multi-Document Summarization

This is my Advanced Machine Learning Assignment 1. It reads related PDFs,
creates text embeddings, groups similar chunks using my K-Means implementation,
and uses local Ollama models to create a final summary.

```text
PDFs -> chunks -> embeddings -> K-Means -> topics -> final summary
```

The agents are in `agents/`, my K-Means code is in `model/KMeans.py`, and the
full process is demonstrated in `assignment1.ipynb`.

## Setup

I created the environment and installed its requirements with:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

I generated `requirements.txt` from the installed packages with:

```bash
uv pip freeze > requirements.txt
```

The project uses `embeddinggemma` for embeddings and `gemma4:e2b` for topic
generation and summarization:

```bash
ollama pull embeddinggemma
ollama pull gemma4:e2b
```

To run the project, open `assignment1.ipynb`, select `.venv/bin/python` as the
kernel, make sure Ollama is running, and run the cells from top to bottom.
