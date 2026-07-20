# 🏦 Smart RAG Document Analyzer

<p align="center">
  <img src="assets/img.png" alt="NeuroSchedule AI Banner" width="600" height="500">
</p>

A privacy-first, fully local Retrieval-Augmented Generation (RAG) system for extracting accurate answers from PDF documents — built with sensitive documents (bank loan statements, financial agreements, contracts) in mind.

No document content is ever sent to an external API. Extraction, retrieval, and answer generation all run on your own machine.

---

## ✨ Key Features

- **🔒 Fully Local / Offline** — No document text is ever transmitted to a third-party service. All processing (PDF extraction, embeddings, retrieval, answer generation) happens on-device.
- **🔍 Hybrid Retrieval** — Combines dense semantic search (FAISS + sentence-transformer embeddings) with sparse keyword search (BM25), fused using Reciprocal Rank Fusion (RRF) for higher retrieval precision than either method alone.
- **🧠 Zero-Hallucination Answer Engine** — Answers are extracted directly from retrieved document text, not generated freely. If the information isn't in the document, the system says so instead of guessing.
- **📊 Structured Document Handling** — Custom extraction logic for form-style `Label : Value` fields and table-flattened data (common in bank/loan PDFs), not just plain prose.
- **⚡ Lightweight** — No GPU required, no external LLM API costs, minimal dependencies.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| PDF Extraction | PyPDF2 |
| Semantic Search | Sentence-Transformers (`all-MiniLM-L6-v2`) + FAISS |
| Keyword Search | BM25 (`rank_bm25`) |
| Answer Extraction | Custom rule-based + BM25-ranked extraction engine |
| Frontend | HTML / CSS / Vanilla JavaScript |
| Templating | Jinja2 |


---

## 🚀 Getting Started

### 1. Clone / download the project
```bash
git clone (https://github.com/dhruvildave235/InfoVault)
cd RAG
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
uvicorn main:app --reload
```

### 5. Open in your browser
```
http://127.0.0.1:8000
```

---

## 📖 How It Works

1. **Upload** — A PDF is uploaded, text is extracted (`utils.py`), and split into overlapping, sentence-aware chunks.
2. **Indexing** — Chunks are embedded and indexed in FAISS (`rag.py`), and a BM25 index is built (`hybrid.py`).
3. **Query** — On asking a question:
   - Semantic search retrieves the top matching chunks by meaning.
   - BM25 retrieves the top matching chunks by exact keyword overlap.
   - Both result sets are fused via Reciprocal Rank Fusion and reranked.
4. **Answer Generation** (`llm.py`) — The engine tries, in order:
   - **Label:Value matching** — for fields like `Loan Amount Sanctioned : Rs. 200000/-`
   - **Proximity extraction** — for table-flattened data like `EMI Amount ... Rs 2546`
   - **Sentence-relevance ranking** — as a general-purpose fallback for prose
   - If none clear a minimum relevance threshold, it returns a clear "not found in document" response rather than fabricating an answer.

---

## 🔐 Privacy Notes

- All processing is local to the machine running the app.
- The uploaded PDF is temporarily saved to `data/sample.pdf` on disk. If working with real sensitive documents, delete this file manually after use, or add automated cleanup.
- This app is intended for local/offline use (`localhost`). If deployed on a public server, standard web security practices (auth, HTTPS, upload limits) should be added — this is a separate concern from the AI pipeline itself, which never calls external services.

---

## ⚠️ Known Limitations

- **Extractive, not generative** — Answers are selected from retrieved text, not synthesized in fluent natural language across multiple facts. This is a deliberate design choice for accuracy and hallucination-avoidance, not a bug.
- **Single-document, single-session state** — The app holds one document in memory at a time; it isn't built for concurrent multi-user document sets.
- **Scanned PDFs** — Documents without selectable text (image-only scans) won't extract without OCR, which isn't currently integrated.

---


## 👤 Author

Built as an academic project exploring hybrid retrieval and privacy-preserving document intelligence for sensitive document domains.
