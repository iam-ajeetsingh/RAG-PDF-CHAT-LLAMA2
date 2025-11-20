"""
Gradio-based remote client for the FastAPI RAG server.
Run this on any machine that can reach the backend server.
"""
import os
from typing import Tuple

import gradio as gr
import requests

BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/query"
UPLOAD_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/upload"
LIST_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/pdfs"
DELETE_ENDPOINT_TEMPLATE = f"{BACKEND_URL.rstrip('/')}/pdfs/{{filename}}"


def ask_question(question: str) -> Tuple[str, str]:
    """Send a question to the backend and return answer + context."""
    if not question.strip():
        return "Please enter a question.", ""
    try:
        resp = requests.post(
            QUERY_ENDPOINT,
            json={"question": question},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "[No answer returned]")
        context_chunks = data.get("context", [])
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "[No context returned]"
        return answer, context_text
    except requests.RequestException as exc:
        return f"[ERROR] {exc}", ""


def upload_pdf(file: gr.File) -> str:
    """Upload a PDF file to the backend for ingestion."""
    if file is None:
        return "No file selected."
    if not file.name.lower().endswith(".pdf"):
        return "Please upload a PDF file."
    try:
        with open(file.name, "rb") as f_in:
            files = {"file": (os.path.basename(file.name), f_in, "application/pdf")}
            resp = requests.post(UPLOAD_ENDPOINT, files=files, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            detail = data.get("detail", "Uploaded successfully.")
            return f"✅ {detail}"
    except requests.RequestException as exc:
        return f"[ERROR] {exc}"


def refresh_pdf_list() -> str:
    """Fetch the list of indexed PDFs."""
    try:
        resp = requests.get(LIST_ENDPOINT, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        pdfs = data.get("pdfs", [])
        if not pdfs:
            return "No PDFs indexed."
        return "\n".join(f"- {pdf_name}" for pdf_name in pdfs)
    except requests.RequestException as exc:
        return f"[ERROR] {exc}"


def delete_pdf(filename: str) -> str:
    """Delete a PDF by filename from the backend index."""
    cleaned = filename.strip()
    if not cleaned:
        return "Please enter a PDF filename to delete."

    endpoint = DELETE_ENDPOINT_TEMPLATE.format(filename=cleaned)
    try:
        resp = requests.delete(endpoint, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        detail = data.get("detail", "Deleted.")
        return f"🗑️ {detail}"
    except requests.HTTPError as exc:
        return f"[ERROR] {exc}"
    except requests.RequestException as exc:
        return f"[ERROR] {exc}"


with gr.Blocks(title="Remote RAG Client") as demo:
    gr.Markdown("## Remote RAG Client\nConnects to a FastAPI backend serving the PDF knowledge base.")

    with gr.Tab("Ask a Question"):
        question = gr.Textbox(label="Question", placeholder="Ask something about your PDFs...", lines=2)
        answer = gr.Textbox(label="Answer", lines=6)
        context = gr.Textbox(label="Context (Top Chunks)", lines=10)
        ask_button = gr.Button("Ask", variant="primary")

        ask_button.click(fn=ask_question, inputs=question, outputs=[answer, context])

    with gr.Tab("Manage PDFs"):
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Upload Status")
        upload_button = gr.Button("Upload & Ingest")
        upload_button.click(fn=upload_pdf, inputs=file_input, outputs=upload_status)

        list_button = gr.Button("Refresh Indexed PDFs")
        pdf_list_output = gr.Textbox(label="Indexed PDFs", lines=8)
        list_button.click(fn=refresh_pdf_list, outputs=pdf_list_output)

        delete_name = gr.Textbox(label="PDF filename to delete (exact match)", lines=1)
        delete_status = gr.Textbox(label="Delete Status")
        delete_button = gr.Button("Delete PDF", variant="stop")
        delete_button.click(fn=delete_pdf, inputs=delete_name, outputs=delete_status)

if __name__ == "__main__":
    demo.launch(share=True)

