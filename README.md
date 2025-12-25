# RAG Document Intelligence (Local)

A modular, local **Retrieval-Augmented Generation (RAG)** pipeline for document intelligence.  
Designed for applied AI engineering with explicit ingestion, chunking, embedding, retrieval, and generation stages.

Runs fully **offline**, **CPU-only**, and is suitable for reproducible experimentation and evaluation.

---

## Prerequisites

- **Python**: 3.11 or 3.12  
- **OS**: Windows / Linux / macOS  
- **Hardware**: 16 GB RAM recommended, no GPU required  

Verify Python installation:
```bash
python --version
```

---

## Setup

```bash
git clone <repo-url>
cd rag-document-intelligence
python -m venv .venv
```

Activate virtual environment:

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -e .
```

---

## Input Data

Place PDF files in:
```
data/raw/
```

---

## Running the Pipeline

The pipeline is intentionally modular. Each stage can be executed independently.

```bash
python scripts/ingest.py
python scripts/chunk.py
python scripts/embed.py
python scripts/query.py
```

The final step starts an **interactive query loop**.

### Retrieval Parameters

The query step exposes two parameters to control retrieval and context selection:

- `--retrieval-k`  
  Number of top-ranked chunks retrieved from the vector index (FAISS) based on embedding similarity.

- `--context-k`  
  Number of chunks selected from the retrieved set and included in the LLM prompt as context.

**Why both exist:**  
Retrieval (`retrieval-k`) is optimized for recall, while context selection (`context-k`) is constrained by LLM context limits and generation quality.

**Example:**

```bash
python scripts/query.py --retrieval-k 8 --context-k 4```
---
## Models

### Embeddings
- `sentence-transformers/all-MiniLM-L6-v2`

### LLM (current)
- `Qwen/Qwen2.5-0.5B-Instruct`  
  *(local, CPU-friendly)*

---

## Design Notes

- No external APIs
- No GPU dependency
- Fully local execution
- Each pipeline stage is independently runnable
- Clear separation of concerns
- Suitable for benchmarking and evaluation workflows

---

## Next Steps

Planned improvements:

- **Performance optimisation**
  - Reduce end-to-end latency
  - Explore quantisation and `llama.cpp`-based runtimes
- **Retrieval quality**
  - Introduce re-rankers (cross-encoder or LLM-based)
  - Experiment with alternative chunking strategies
- **Evaluation**
  - Test multiple LLMs against a curated Q&A dataset
  - Comparative analysis across embeddings and retrieval parameters

---
