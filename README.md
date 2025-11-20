# RAG PDF Chat Application

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Retrieval-Augmented Generation (RAG) application that allows you to query your PDF documents using a local LLM (Llama 2) running on your GPU.

## Features

- 📄 **PDF Upload & Management**: Upload, delete, and manage PDF documents directly from the GUI
- 🔍 **Intelligent Search**: Query your PDF knowledge base using natural language
- 🤖 **Local LLM**: Uses Llama 2 via Ollama running on your local GPU (RTX 4060)
- 💾 **Vector Database**: ChromaDB for efficient semantic search and retrieval
- 🎨 **User-Friendly GUI**: Simple Tkinter interface for easy interaction
- ⚡ **Automatic Ingestion**: Automatically processes and indexes PDFs when uploaded

## Architecture

```
User Question → Embedding → Vector Search (ChromaDB) → Top-K Context Chunks
                                                              ↓
                        Answer ← LLM (Llama 2 via Ollama) ← Prompt (Context + Question)
```

## Prerequisites

1. **Python 3.10+** with Conda
2. **Ollama** installed and running with Llama 2 model
3. **NVIDIA GPU** (RTX 4060) with CUDA support
4. **Anaconda/Miniconda** for environment management

## Installation

### 1. Create Conda Environment

```bash
conda create -n rag-pipeline python=3.10
conda activate rag-pipeline
```

### 2. Install PyTorch (GPU Support)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama

1. Download Ollama from [https://ollama.com/download](https://ollama.com/download)
2. Install Ollama on your system
3. Pull Llama 2 model:

```bash
ollama pull llama2
```

### 5. Start Ollama Service

In a separate terminal, start Ollama:

```bash
ollama serve
```

Keep this running while using the application.

## Usage

### GUI Application (Recommended)

Launch the end-to-end GUI application:

```bash
python rag_app.py
```

**Features:**
- **Upload PDF**: Click "Upload PDF" to add new documents
- **Delete PDF**: Select a PDF from the list and click "Delete Selected" to remove it
- **Re-ingest All**: Re-process all PDFs in the `sample_pdfs/` directory
- **Ask Questions**: Type your question and click "Ask" or press Enter
- **View Context**: Switch to the "Context" tab to see which chunks were used

### Command-Line Interface

Query your knowledge base from the command line:

```bash
python rag_query.py "What is RAG?"
```

### Manual PDF Ingestion

To manually ingest PDFs from the `sample_pdfs/` directory:

```bash
python ingest_pdfs.py
```

## Remote Access (FastAPI + Gradio)

Run the heavy RAG pipeline on this machine while controlling it from another laptop:

1. **Start the FastAPI backend**
   ```bash
   uvicorn rag_server:app --host 0.0.0.0 --port 8000
   ```
   | Environment Variable | Default                                   | Description                           |
   |----------------------|-------------------------------------------|---------------------------------------|
   | `OLLAMA_URL`        | `http://localhost:11434/api/generate`     | URL of your Ollama instance           |
   | `OLLAMA_MODEL`      | `llama2`                                  | Model tag to query                    |
   | `RAG_PDF_DIR`       | `sample_pdfs`                             | Directory containing PDFs             |
   | `RAG_QUERY_TOP_K`   | `3`                                       | Context chunks retrieved per query    |

2. **Open firewall / note LAN IP** so remote devices can reach `http://<server-ip>:8000`.

3. **Launch the Gradio client** on the remote laptop:
   ```bash
   export RAG_BACKEND_URL="http://SERVER_IP:8000"
   python rag_gradio_client.py
   ```
   This client lets you ask questions and upload/list PDFs through your browser.

## Containerized Deployment (Docker + Compose)

Deploy both the FastAPI backend and Ollama with Docker for repeatable setups:

1. **Start services**
   ```bash
   docker compose up --build
   ```
   - `rag-server` (FastAPI) exposed on `8000`
   - `ollama` exposed on `11434`
   - `ollama-init` automatically pulls the llama2 model on first startup

2. **If model doesn't auto-pull** (manual pull)
   
   On Windows:
   ```bash
   pull_model.bat
   ```
   
   On Linux/Mac:
   ```bash
   chmod +x pull_model.sh
   ./pull_model.sh
   ```
   
   Or manually:
   ```bash
   docker exec rag_ollama ollama pull llama2
   ```

3. **Persist data**
   - `./sample_pdfs` → `/data/pdfs`
   - `./chromadb_store` → `/data/chromadb`
   - `./ollama` → `/root/.ollama` (model cache)

4. **Configure**
   Override environment variables in `docker-compose.yml` or via `.env`:
   `OLLAMA_MODEL`, `RAG_PDF_DIR`, `CHROMA_PATH`, `RAG_QUERY_TOP_K`, etc.

5. **Access**
   - API: `http://localhost:8000`
   - Point Gradio or other clients to `http://<host-ip>:8000`

6. **Performance Notes**
   - **First query is slow** (~40-50 seconds): The model needs to load into GPU memory
   - **Subsequent queries are fast** (~1-5 seconds): Model stays loaded in GPU memory
   - This is normal behavior - Ollama keeps the model in GPU memory for faster inference

## Project Structure

```
.
├── rag_server.py      # FastAPI backend for remote access
├── rag_gradio_client.py # Gradio UI client for remote use
├── rag_app.py          # Main GUI application (end-to-end)
├── rag_gui.py          # Simple GUI (original version)
├── rag_query.py        # Command-line query tool
├── rag_utils.py        # Utility functions for PDF processing and RAG
├── ingest_pdfs.py      # Manual PDF ingestion script
├── generate_pdfs.py    # Script to generate sample PDFs
├── requirements.txt    # Python dependencies
├── sample_pdfs/        # Directory for PDF files
├── chromadb_store/     # ChromaDB persistent storage
└── README.md           # This file
```

## Configuration

You can modify these settings in `rag_utils.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1500 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 400 characters)
- `QUERY_TOP_K`: Number of context chunks to retrieve (default: 3)
- `CHROMA_COLLECTION`: Name of the ChromaDB collection (default: 'rag_pdf_collection')

You can modify these settings in `rag_app.py`:

- `OLLAMA_URL`: Ollama API endpoint (default: 'http://localhost:11434/api/generate')
- `MODEL_NAME`: Ollama model to use (default: 'llama2')
- `PDF_DIR`: Directory for PDF files (default: 'sample_pdfs')

## How It Works

1. **PDF Ingestion**:
   - PDFs are read and text is extracted
   - Text is chunked into overlapping segments
   - Each chunk is embedded using sentence-transformers
   - Embeddings are stored in ChromaDB with metadata

2. **Query Processing**:
   - User question is embedded using the same model
   - Vector similarity search finds top-K relevant chunks
   - Context chunks are assembled into a prompt
   - Llama 2 generates an answer based on the context

3. **Answer Generation**:
   - LLM receives context + question
   - LLM generates answer using only the provided context
   - Answer is displayed in the GUI or CLI

## Troubleshooting

### Ollama Not Found

If you get `'ollama' is not recognized`, make sure:
- Ollama is installed and added to your system PATH
- Restart your terminal after installing Ollama

### Collection Not Found

If you get `Collection does not exist`:
- Run `python ingest_pdfs.py` first to create the collection
- Or use the GUI "Re-ingest All" button

### GPU Not Used

To verify PyTorch is using your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should print True
```

### Ollama Connection Error

Make sure Ollama is running:
```bash
ollama serve
```

Or check if Ollama is accessible:
```bash
curl http://localhost:11434/api/tags
```

### Docker: Model Not Found (404 error)

If you get `404` errors when querying via Docker:
- The `llama2` model needs to be pulled into the container
- The `ollama-init` service should do this automatically, but if it fails:
  ```bash
  docker exec rag_ollama ollama pull llama2
  ```
- Or run `pull_model.bat` (Windows) or `./pull_model.sh` (Linux/Mac)

### Docker: First Query is Slow

**Normal behavior:**
- First query takes ~40-50 seconds: Model must load from disk into GPU memory
- Subsequent queries are fast (~1-5 seconds): Model stays loaded in GPU memory
- This is expected - Ollama keeps the model in GPU memory for faster inference
- The model stays loaded until the container restarts or memory is freed

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages:
- `torch` (with CUDA support)
- `transformers`
- `sentence-transformers`
- `chromadb`
- `pypdf`
- `requests`
- `tkinter` (usually comes with Python)

## License

This project is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

