"""
End-to-end RAG Application with PDF Management GUI.
"""
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
from threading import Thread
import requests
import os
import shutil
from typing import Callable, Optional, List

from rag_utils import (
    ingest_pdf, ingest_directory, delete_pdf, get_indexed_pdfs,
    query_rag, CHROMA_COLLECTION, QUERY_TOP_K
)

# Configuration
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama2'
PDF_DIR = 'sample_pdfs'


class RAGApplication:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RAG PDF Chat - End-to-End Application")
        self.root.geometry('900x700')
        
        # Status tracking
        self.is_ingesting = False
        self.is_querying = False
        
        self.setup_ui()
        self.refresh_pdf_list()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Top frame for PDF management
        pdf_frame = tk.LabelFrame(self.root, text="PDF Management", padx=10, pady=10)
        pdf_frame.pack(fill='x', padx=10, pady=5)
        
        # PDF list with scrollbar
        list_frame = tk.Frame(pdf_frame)
        list_frame.pack(fill='both', expand=True, pady=5)
        
        self.pdf_listbox = tk.Listbox(list_frame, height=4, font=("Arial", 10))
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.pdf_listbox.yview)
        self.pdf_listbox.config(yscrollcommand=scrollbar.set)
        self.pdf_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # PDF management buttons
        btn_frame = tk.Frame(pdf_frame)
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="Upload PDF", command=self.upload_pdf,
                 font=("Arial", 10), bg="#4CAF50", fg="white").pack(side='left', padx=5)
        tk.Button(btn_frame, text="Delete Selected", command=self.delete_selected_pdf,
                 font=("Arial", 10), bg="#f44336", fg="white").pack(side='left', padx=5)
        tk.Button(btn_frame, text="Re-ingest All", command=self.reingest_all,
                 font=("Arial", 10), bg="#2196F3", fg="white").pack(side='left', padx=5)
        tk.Button(btn_frame, text="Refresh List", command=self.refresh_pdf_list,
                 font=("Arial", 10)).pack(side='left', padx=5)
        
        # Status label
        self.status_label = tk.Label(pdf_frame, text="Ready", fg="green", font=("Arial", 9))
        self.status_label.pack(pady=5)
        
        # Query frame
        query_frame = tk.LabelFrame(self.root, text="Ask Question", padx=10, pady=10)
        query_frame.pack(fill='x', padx=10, pady=5)
        
        entry_frame = tk.Frame(query_frame)
        entry_frame.pack(fill='x', pady=5)
        
        tk.Label(entry_frame, text="Question:", font=("Arial", 11)).pack(side='left', padx=5)
        self.query_entry = tk.Entry(entry_frame, font=("Arial", 12))
        self.query_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.query_entry.bind('<Return>', lambda e: self.ask_question())
        
        tk.Button(entry_frame, text="Ask", command=self.ask_question,
                 font=("Arial", 12), bg="#9C27B0", fg="white", width=10).pack(side='left', padx=5)
        
        # Results frame with notebook (tabs)
        results_frame = tk.LabelFrame(self.root, text="Results", padx=10, pady=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill='both', expand=True)
        
        # Answer tab
        answer_frame = tk.Frame(notebook)
        notebook.add(answer_frame, text="Answer")
        self.answer_area = scrolledtext.ScrolledText(
            answer_frame, wrap=tk.WORD, font=("Arial", 11), bg="#ffffff"
        )
        self.answer_area.pack(fill='both', expand=True)
        self.answer_area.insert(tk.END, "Enter a question and click 'Ask' to get an answer.\n")
        
        # Context tab
        context_frame = tk.Frame(notebook)
        notebook.add(context_frame, text="Context (Top 3 Chunks)")
        self.context_area = scrolledtext.ScrolledText(
            context_frame, wrap=tk.WORD, font=("Arial", 10), bg="#f9f5e3"
        )
        self.context_area.pack(fill='both', expand=True)
        self.context_area.insert(tk.END, "[Context chunks will appear here]\n")
    
    def update_status(self, message: str, color: str = "black"):
        """Update status label."""
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()
    
    def refresh_pdf_list(self):
        """Refresh the list of indexed PDFs."""
        self.pdf_listbox.delete(0, tk.END)
        pdfs = get_indexed_pdfs()
        if pdfs:
            for pdf in pdfs:
                self.pdf_listbox.insert(tk.END, pdf)
        else:
            self.pdf_listbox.insert(tk.END, "[No PDFs indexed. Upload a PDF to start.]")
        self.update_status(f"Found {len(pdfs)} indexed PDF(s)", "green")
    
    def upload_pdf(self):
        """Upload and ingest a new PDF file."""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        filename = os.path.basename(file_path)
        dest_path = os.path.join(PDF_DIR, filename)
        
        # Ensure PDF directory exists
        os.makedirs(PDF_DIR, exist_ok=True)
        
        # Copy file to PDF directory
        try:
            shutil.copy2(file_path, dest_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy file: {str(e)}")
            return
        
        # Ingest the PDF
        self.update_status(f"Ingesting {filename}...", "blue")
        self.is_ingesting = True
        
        def ingest_thread():
            success, message, num_chunks = ingest_pdf(dest_path, PDF_DIR)
            self.root.after(0, lambda: self.on_ingest_complete(success, message, num_chunks))
        
        Thread(target=ingest_thread, daemon=True).start()
    
    def on_ingest_complete(self, success: bool, message: str, num_chunks: int):
        """Handle completion of PDF ingestion."""
        self.is_ingesting = False
        if success:
            self.update_status(message, "green")
            self.refresh_pdf_list()
            messagebox.showinfo("Success", f"{message}\nCreated {num_chunks} chunks.")
        else:
            self.update_status(message, "red")
            messagebox.showerror("Error", message)
    
    def delete_selected_pdf(self):
        """Delete the selected PDF from index."""
        selection = self.pdf_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a PDF to delete.")
            return
        
        filename = self.pdf_listbox.get(selection[0])
        if "[No PDFs indexed" in filename:
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", f"Delete {filename} from index?"):
            return
        
        # Delete from ChromaDB
        success, message = delete_pdf(filename)
        
        if success:
            self.update_status(message, "green")
            self.refresh_pdf_list()
            messagebox.showinfo("Success", message)
        else:
            self.update_status(message, "red")
            messagebox.showerror("Error", message)
    
    def reingest_all(self):
        """Re-ingest all PDFs from the directory."""
        if not messagebox.askyesno("Confirm", "Re-ingest all PDFs? This may take a moment."):
            return
        
        self.update_status("Re-ingesting all PDFs...", "blue")
        self.is_ingesting = True
        
        def ingest_thread():
            success_count, failure_count, messages = ingest_directory(PDF_DIR)
            summary = f"Success: {success_count}, Failed: {failure_count}"
            self.root.after(0, lambda: self.on_reingest_complete(summary, messages))
        
        Thread(target=ingest_thread, daemon=True).start()
    
    def on_reingest_complete(self, summary: str, messages: List[str]):
        """Handle completion of re-ingestion."""
        self.is_ingesting = False
        self.update_status(summary, "green")
        self.refresh_pdf_list()
        messagebox.showinfo("Re-ingestion Complete", f"{summary}\n\n" + "\n".join(messages[:5]))
    
    def ask_question(self):
        """Query the RAG system with the user's question."""
        question = self.query_entry.get().strip()
        if not question:
            messagebox.showwarning("Empty Question", "Please enter a question.")
            return
        
        if self.is_querying:
            messagebox.showinfo("Processing", "Please wait for the current query to complete.")
            return
        
        # Clear previous results
        self.answer_area.delete('1.0', tk.END)
        self.answer_area.insert(tk.END, "Processing... This may take a moment.\n")
        self.context_area.delete('1.0', tk.END)
        self.context_area.insert(tk.END, "[Loading context...]\n")
        
        self.update_status("Querying RAG system...", "blue")
        self.is_querying = True
        
        def query_thread():
            try:
                # Get context from RAG
                _, context_chunks, metadatas = query_rag(question, QUERY_TOP_K)
                context = '\n'.join(context_chunks) if context_chunks else "[No context found]"
                
                # Build prompt for LLM
                prompt = f"""You are an AI assistant with access to the following document excerpts:\n\n{context}\n\nUsing ONLY this information, answer the following question as accurately as possible:\n\nQuestion: {question}\nAnswer:"""
                
                # Query Ollama
                payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
                resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
                data = resp.json()
                answer = data.get('response', '[No answer received]').strip()
                
                # Format context with sources
                context_with_sources = ""
                for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas), 1):
                    source = meta.get('source', 'Unknown') if meta else 'Unknown'
                    context_with_sources += f"[{i}] Source: {source}\n{chunk}\n\n{'='*50}\n\n"
                
                self.root.after(0, lambda: self.on_query_complete(answer, context_with_sources, True, None))
            except requests.exceptions.RequestException as e:
                error_msg = f"Error connecting to Ollama. Make sure Ollama is running.\n{str(e)}"
                self.root.after(0, lambda: self.on_query_complete("", "", False, error_msg))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self.on_query_complete("", "", False, error_msg))
        
        Thread(target=query_thread, daemon=True).start()
    
    def on_query_complete(self, answer: str, context: str, success: bool, error: Optional[str]):
        """Handle completion of RAG query."""
        self.is_querying = False
        
        if success:
            self.answer_area.delete('1.0', tk.END)
            self.answer_area.insert(tk.END, answer + '\n')
            self.context_area.delete('1.0', tk.END)
            self.context_area.insert(tk.END, context if context else "[No context found]\n")
            self.update_status("Query complete", "green")
        else:
            self.answer_area.delete('1.0', tk.END)
            self.answer_area.insert(tk.END, error or "An error occurred.")
            self.update_status("Query failed", "red")
            messagebox.showerror("Error", error or "An error occurred while processing your query.")


def main():
    """Main entry point for the application."""
    # Check if Ollama is running
    try:
        requests.get('http://localhost:11434/api/tags', timeout=2)
    except requests.exceptions.RequestException:
        result = messagebox.askokcancel(
            "Ollama Not Running",
            "Could not connect to Ollama. Make sure Ollama is running.\n\nContinue anyway?"
        )
        if not result:
            return
    
    root = tk.Tk()
    app = RAGApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()

