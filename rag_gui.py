import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from sentence_transformers import SentenceTransformer
import chromadb
import requests

CHROMA_COLLECTION = 'rag_pdf_collection'
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama2'
CHROMA_PATH = 'chromadb_store'
QUERY_TOP_K = 3

# Load models once
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=CHROMA_PATH)
existing = [c.name for c in client.list_collections()]
if CHROMA_COLLECTION in existing:
    collection = client.get_collection(CHROMA_COLLECTION)
else:
    raise ValueError(f"Collection '{CHROMA_COLLECTION}' not found. Run ingest_pdfs.py first.")

def run_query(question, update_fn):
    query_emb = embedder.encode([question])[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=QUERY_TOP_K,
        include=['documents', 'metadatas']
    )
    context_chunks = results['documents'][0]
    context = '\n'.join(context_chunks)
    prompt = f"""
    You are an AI assistant with access to the following document excerpts:\n\n
    {context}\n\n
    Using ONLY this information, answer the following question as accurately as possible:\n\n
    Question: {question}\n
    Answer:"""
    
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = resp.json()
        answer = data.get('response', '[No answer received]')
    except Exception as e:
        answer = "[ERROR] " + str(e)
    update_fn(answer.strip(), context.strip())

# GUI
root = tk.Tk()
root.title("RAG PDF Chat")
root.geometry('700x500')

frame = tk.Frame(root)
frame.pack(pady=10)

lbl = tk.Label(frame, text="Ask a question based on your PDFs:")
lbl.pack()

entry = tk.Entry(frame, width=60, font=("Arial", 14))
entry.pack(padx=10, pady=5)

result_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=12, font=("Arial", 12))
result_area.pack(padx=10, pady=5, fill='both', expand=True)

context_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=12, font=("Arial", 12), bg="#f9f5e3")
context_area.pack(padx=10, pady=5, fill='both', expand=True)
context_area.insert(tk.END, "\n[Context Chunks Will Show Here]")
context_area.config(state='disabled')

def ask():
    q = entry.get().strip()
    if not q:
        result_area.insert(tk.END, "\n[Please enter a question.]\n")
        return
    result_area.delete('1.0', tk.END)
    result_area.insert(tk.END, "Answering... This may take a moment.\n")
    context_area.config(state='normal')
    context_area.delete('1.0', tk.END)
    context_area.insert(tk.END, "[Loading context...]")
    context_area.config(state='disabled')

    def update(answer, context):
        result_area.delete('1.0', tk.END)
        result_area.insert(tk.END, answer + '\n')
        context_area.config(state='normal')
        context_area.delete('1.0', tk.END)
        context_area.insert(tk.END, context)
        context_area.config(state='disabled')

    Thread(target=run_query, args=(q, update), daemon=True).start()

btn = tk.Button(frame, text="Ask", command=ask, font=("Arial", 12))
btn.pack(pady=5)

root.mainloop()
