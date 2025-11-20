import os
from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Parameters
PDF_DIR = 'sample_pdfs'
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
CHROMA_COLLECTION = 'rag_pdf_collection'

# 1. Extract PDF text
def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + '\n'
    return text.strip()

# 2. Chunk text
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 3. Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast

# 4. ChromaDB setup
client = chromadb.PersistentClient(path="chromadb_store")
if CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
    collection = client.get_collection(CHROMA_COLLECTION)
else:
    collection = client.create_collection(CHROMA_COLLECTION)

# 5. Ingest all PDFs
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(PDF_DIR, filename)
        print(f"Processing {filename}...")
        text = read_pdf(file_path)
        print("--------------------------------------------------")
        print(text)
        print("--------------------------------------------------")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        print("--------------------------------------------------")
        print(chunks)
        print("--------------------------------------------------")
        embeddings = embedder.encode(chunks)
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

        # Store in ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        print(f"Stored {len(chunks)} chunks from {filename}.")

print("Ingestion complete! Chunks are now in ChromaDB.")
