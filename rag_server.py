"""
FastAPI server exposing the RAG pipeline over HTTP so remote clients (e.g. Gradio UI)
can interact with the knowledge base and Llama 2/Ollama backend.
"""
from __future__ import annotations

import os
import shutil
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_utils import (
    ingest_pdf,
    ingest_directory,
    delete_pdf,
    get_indexed_pdfs,
    query_rag,
    QUERY_TOP_K,
)

# Configuration (override via environment variables if needed)
PDF_DIR = os.environ.get("RAG_PDF_DIR", "sample_pdfs")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama2")
DEFAULT_TOP_K = int(os.environ.get("RAG_QUERY_TOP_K", QUERY_TOP_K))


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    sources: List[str]


class StatusResponse(BaseModel):
    status: str
    detail: str


class PDFListResponse(BaseModel):
    pdfs: List[str]


class ReingestResponse(BaseModel):
    success_count: int
    failure_count: int
    messages: List[str]


def ensure_pdf_dir() -> str:
    """Ensure the PDF directory exists and return its path."""
    os.makedirs(PDF_DIR, exist_ok=True)
    return PDF_DIR


app = FastAPI(title="RAG PDF Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=StatusResponse)
def health_check():
    """Simple health endpoint."""
    return StatusResponse(status="ok", detail="Server is running")


@app.get("/pdfs", response_model=PDFListResponse)
def list_pdfs():
    """List all indexed PDFs."""
    pdfs = get_indexed_pdfs()
    return PDFListResponse(pdfs=pdfs)


@app.post("/upload", response_model=StatusResponse)
def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    ensure_pdf_dir()
    dest_path = os.path.join(PDF_DIR, file.filename)

    try:
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc
    finally:
        file.file.close()

    success, message, _ = ingest_pdf(dest_path, PDF_DIR)
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return StatusResponse(status="success", detail=message)


@app.delete("/pdfs/{filename}", response_model=StatusResponse)
def delete_pdf_endpoint(filename: str):
    """Delete all chunks for a given PDF."""
    success, message = delete_pdf(filename)
    if not success:
        raise HTTPException(status_code=404, detail=message)
    return StatusResponse(status="success", detail=message)


@app.post("/reingest", response_model=ReingestResponse)
def reingest_all_endpoint():
    """Re-ingest all PDFs from the directory."""
    ensure_pdf_dir()
    success_count, failure_count, messages = ingest_directory(PDF_DIR)
    return ReingestResponse(
        success_count=success_count,
        failure_count=failure_count,
        messages=messages,
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    """Answer a question using the RAG pipeline and Ollama."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    top_k = payload.top_k or DEFAULT_TOP_K

    try:
        _, context_chunks, metadatas = query_rag(question, top_k)
        if not context_chunks:
            return QueryResponse(
                answer="Could not find relevant context for the question.",
                context=[],
                sources=[],
            )

        context_text = "\n".join(context_chunks)
        prompt = (
            "You are an AI assistant with access to the following document excerpts:\n\n"
            f"{context_text}\n\n"
            "Using ONLY this information, answer the following question as accurately as possible.\n\n"
            f"Question: {question}\nAnswer:"
        )

        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "[No answer received]").strip()

        sources = []
        for meta in metadatas:
            src = meta.get("source", "Unknown") if meta else "Unknown"
            sources.append(src)

        return QueryResponse(answer=answer, context=context_chunks, sources=sources)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Ollama at {OLLAMA_URL}: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=False)

