# 🔬 AI Research Assistant

A **production-grade Retrieval-Augmented Generation (RAG)** system for intelligent PDF document analysis. Ask questions, get answers with citations, explore document insights — all running locally with no API keys required.

---

## Architecture

```
User Query
    │
    ▼
Query Expansion (keyword variants, definition/explanation phrasings)
    │
    ├─────────────────────┐
    ▼                     ▼
FAISS Semantic         BM25 Keyword
Search                 Search
    │                     │
    └──────────┬──────────┘
               ▼
        Merge + Deduplicate
               │
               ▼
        Re-rank (cosine similarity)
               │
               ▼
        Context Compression
        (sentence-level filtering)
               │
               ▼
        LLM Generation
        (google/flan-t5-base)
               │
               ▼
        Answer + Source Citations
```

---

## Project Structure

```
ai-research-assistant/
│
├── data/
│   └── pdfs/                  ← Drop your PDFs here
│
├── vector_store/              ← FAISS index (auto-generated)
├── logs/                      ← Ingestion / retrieval / RAG logs
│
├── ingest.py                  ← PDF loading, chunking, FAISS embedding
├── retriever.py               ← Hybrid FAISS+BM25, re-rank, compress
├── rag_pipeline.py            ← LLM chain with citations
├── document_insights.py       ← Summary, topics, concepts
├── evaluation.py              ← Latency & accuracy metrics
├── app.py                     ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component         | Library / Model                        |
|-------------------|----------------------------------------|
| Orchestration     | LangChain                              |
| Embeddings        | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store      | FAISS (CPU)                            |
| Keyword Search    | BM25 (rank_bm25)                       |
| LLM               | `google/flan-t5-base` (HuggingFace)    |
| PDF Parsing       | PyPDFLoader (pypdf)                    |
| UI                | Streamlit                              |
| ML Utilities      | scikit-learn, numpy, nltk              |

---

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads ~2GB of model weights (flan-t5, all-MiniLM-L6-v2). Subsequent runs use the local cache.

### 3. Add your PDFs

Copy PDF files into `data/pdfs/`:
```
data/pdfs/RL_Book.pdf
data/pdfs/LectureNotes.pdf
```

### 4. Run ingestion

```bash
python ingest.py
```

This loads PDFs → splits into chunks → embeds → saves FAISS index to `vector_store/`.

To ingest a single file:
```bash
python ingest.py --file data/pdfs/MyPaper.pdf
```

### 5. Launch the app

```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`.

---

## Features

### 💬 Chat Tab
- Conversational Q&A with full chat history
- Source citation cards (document name, page, relevance score)
- Live latency and citation count metrics

### 🔍 Search Tab
- Hybrid retrieval (FAISS + BM25) without LLM generation
- Configurable top-k (3, 5, 8, 10)
- Shows raw ranked passages with scores

### 📊 Insights Tab
- **Automatic summary** (extractive, frequency-scored sentences)
- **Key topics** (TF-IDF unigrams + bigrams)
- **Key concepts** (frequency-ranked meaningful tokens)

### 📤 Sidebar Upload
- Drag-and-drop PDF upload directly from the UI
- Automatically ingests and indexes the new document
- Re-index All button to rebuild the full vector store

---

## Evaluation

Run the evaluation suite against the indexed documents:

```bash
# Retrieval only (fast)
python evaluation.py --no-rag

# Full RAG evaluation
python evaluation.py

# Single query
python evaluation.py --query "What is gradient descent?"
```

Outputs metrics to console and saves `logs/evaluation_report.json`.

| Metric                | Description                              |
|-----------------------|------------------------------------------|
| Retrieval latency     | Time to fetch and re-rank chunks (s)     |
| Avg rerank score      | Mean cosine similarity of top chunks     |
| Chunks retrieved      | Number of chunks returned                |
| RAG latency           | End-to-end query→answer time (s)         |
| Context tokens        | Estimated tokens sent to LLM             |

---

## RAG Pipeline In Detail

### 1. Query Expansion
The raw query is expanded into 3–4 variants:
- Original query
- Keyword-only (stop words removed)
- `"definition of <core phrase>"`
- `"explain <core phrase>"`

All variants are searched in parallel, boosting recall.

### 2. Hybrid Retrieval
- **FAISS**: dense semantic search over `all-MiniLM-L6-v2` embeddings
- **BM25**: sparse keyword search (BM25Okapi) over the same document corpus
- Results merged and deduplicated

### 3. Re-ranking
All candidates are re-scored by cosine similarity between query embedding and chunk embedding. Sorted descending; scores below threshold are dropped.

### 4. Context Compression
Each chunk's individual sentences are scored against the query. Only the top 60% most relevant sentences are kept, reducing noise and prompt length.

### 5. LLM Generation
A `text2text-generation` pipeline with `google/flan-t5-base` receives a structured prompt:
```
Context:
[source chunks]

Question: <user query>

Answer:
```
Output is returned with source citations (document name, page, chunk preview, score).

---

## Example Queries

```
What is reinforcement learning?
Explain the difference between supervised and unsupervised learning.
What are the main challenges in training deep neural networks?
How does the attention mechanism work in transformers?
What is gradient descent and how does it work?
```

---

## Logs

All pipeline steps are logged to the `logs/` directory:

| File                       | Contents                    |
|----------------------------|-----------------------------|
| `logs/ingest.log`          | PDF loading & embedding     |
| `logs/retriever.log`       | Search, merge, re-rank      |
| `logs/rag_pipeline.log`    | Prompt & LLM generation     |
| `logs/insights.log`        | Document analysis           |
| `logs/app.log`             | User queries via Streamlit  |
| `logs/evaluation_report.json` | Evaluation results       |

---

## Swapping the LLM

To use OpenAI or Groq instead of the local flan-t5 model, replace the pipeline in `rag_pipeline.py`:

```python
# OpenAI example
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

Set your API key in a `.env` file:
```
OPENAI_API_KEY=sk-...
```
