import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils import extract_text, smart_chunk
from rag import create_index, retrieve, rerank
from hybrid import build_bm25, keyword_search, reciprocal_rank_fusion
from llm import generate_answer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DATA_DIR = "data"
UPLOAD_PATH = os.path.join(DATA_DIR, "sample.pdf")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


chunks = []
index = None
bm25 = None


# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global chunks, index, bm25

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(DATA_DIR, exist_ok=True)

    size = 0
    try:
        with open(UPLOAD_PATH, "wb") as buffer:
            while chunk_bytes := await file.read(1024 * 1024):
                size += len(chunk_bytes)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="File too large (max 25 MB).",
                    )
                buffer.write(chunk_bytes)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to save uploaded file.")

    try:
        text = extract_text(UPLOAD_PATH)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text found in this PDF. It may be a scanned "
                "image without OCR — try a text-based PDF instead."
            ),
        )

    new_chunks = smart_chunk(text)
    if not new_chunks:
        raise HTTPException(status_code=422, detail="Document produced no usable chunks.")

    try:
        new_index, _ = create_index(new_chunks)
        new_bm25 = build_bm25(new_chunks)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to index the document.")

    # Only commit to global state once every step has succeeded, so a
    # failed upload never leaves the app answering from a stale document.
    chunks, index, bm25 = new_chunks, new_index, new_bm25

    return {"message": "PDF processed successfully", "chunks": len(chunks)}


@app.post("/ask")
async def ask(query: str = Form(...)):
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if index is None or not chunks:
        return {
            "query": query,
            "baseline": [],
            "optimized": [],
            "final_answer": "⚠️ Please upload a PDF document first before asking questions.",
        }

    # Dense (semantic) retrieval.
    baseline = retrieve(query, chunks, index, k=5)

    # Sparse (keyword/BM25) retrieval.
    keyword_results = keyword_search(query, bm25, chunks, k=5)

    # Fuse both ranked lists with Reciprocal Rank Fusion instead of the
    # original `list(set(...))`, which discarded rank order and made
    # the subsequent rerank step start from an unordered, arbitrary set.
    fused = reciprocal_rank_fusion([baseline, keyword_results], top_n=10)

    # Cross-encoder-free semantic rerank over the fused candidate set,
    # batched for speed (see rag.py).
    optimized = rerank(query, fused)[:5]

    if not optimized:
        return {
            "query": query,
            "baseline": baseline,
            "optimized": [],
            "final_answer": (
                "I couldn't find anything relevant to that question in the "
                "uploaded document."
            ),
        }

    answer = generate_answer(query, optimized)

    return {
        "query": query,
        "baseline": baseline,
        "optimized": optimized,
        "final_answer": answer,
    }
