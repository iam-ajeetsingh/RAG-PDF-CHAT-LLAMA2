"""
Utility functions for RAG pipeline: PDF processing, ingestion, and querying.
"""
import os
from typing import List, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration (environment overrides for container use)
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 400))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "rag_pdf_collection")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chromadb_store")
QUERY_TOP_K = int(os.environ.get("RAG_QUERY_TOP_K", 3))
DEFAULT_PDF_DIR = os.environ.get("RAG_PDF_DIR", "sample_pdfs")

# Initialize models (lazy loading)
_embedder = None
_client = None
_collection = None


def get_embedder() -> SentenceTransformer:
    """Get or create the sentence transformer embedder."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def get_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    global _collection
    if _collection is None:
        client = get_client()
        existing = [c.name for c in client.list_collections()]
        if CHROMA_COLLECTION in existing:
            _collection = client.get_collection(CHROMA_COLLECTION)
        else:
            _collection = client.create_collection(CHROMA_COLLECTION)
    return _collection


def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + '\n'
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF {file_path}: {str(e)}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunk text into overlapping segments."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_pdf(file_path: str, pdf_dir: str = DEFAULT_PDF_DIR) -> Tuple[bool, str, int]:
    """
    Ingest a single PDF file into ChromaDB.
    Returns: (success, message, num_chunks)
    """
    try:
        collection = get_collection()
        embedder = get_embedder()
        filename = os.path.basename(file_path)
        
        # Extract and chunk text
        text = read_pdf(file_path)
        if not text:
            return False, f"No text extracted from {filename}", 0
        
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return False, f"No chunks created from {filename}", 0
        
        # Remove existing chunks for this file (for re-ingestion)
        existing_ids = collection.get(
            where={"source": filename},
            include=[]
        )['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
        
        # Create embeddings and store
        embeddings = embedder.encode(chunks)
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        return True, f"Successfully ingested {filename}", len(chunks)
    except Exception as e:
        return False, f"Error ingesting {filename}: {str(e)}", 0


def ingest_directory(pdf_dir: str = DEFAULT_PDF_DIR) -> Tuple[int, int, List[str]]:
    """
    Ingest all PDFs from a directory.
    Returns: (success_count, failure_count, messages)
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        return 0, 0, ["Directory created. Add PDF files to ingest."]
    
    messages = []
    success_count = 0
    failure_count = 0
    
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            success, message, num_chunks = ingest_pdf(file_path, pdf_dir)
            messages.append(message)
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count, messages


def delete_pdf(filename: str) -> Tuple[bool, str]:
    """Delete all chunks for a PDF file from ChromaDB."""
    try:
        collection = get_collection()
        existing_ids = collection.get(
            where={"source": filename},
            include=[]
        )['ids']
        
        if existing_ids:
            collection.delete(ids=existing_ids)
            return True, f"Deleted {len(existing_ids)} chunks for {filename}"
        else:
            return False, f"No chunks found for {filename}"
    except Exception as e:
        return False, f"Error deleting {filename}: {str(e)}"


def get_indexed_pdfs() -> List[str]:
    """Get list of PDF filenames currently indexed in ChromaDB."""
    try:
        collection = get_collection()
        results = collection.get(include=['metadatas'])
        sources = set()
        for metadata in results['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        return sorted(list(sources))
    except Exception:
        return []


def query_rag(question: str, top_k: int = QUERY_TOP_K) -> Tuple[str, List[str], List[dict]]:
    """
    Query the RAG system.
    Returns: (answer, context_chunks, metadata_list)
    """
    try:
        embedder = get_embedder()
        collection = get_collection()
        
        query_emb = embedder.encode([question])[0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        
        context_chunks = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        return None, context_chunks, metadatas
    except Exception as e:
        raise ValueError(f"Error querying RAG: {str(e)}")

