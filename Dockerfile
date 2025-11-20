FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_PDF_DIR=/data/pdfs \
    CHROMA_PATH=/data/chromadb \
    RAG_QUERY_TOP_K=3 \
    OLLAMA_URL=http://ollama:11434/api/generate \
    OLLAMA_MODEL=llama2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8000"]

