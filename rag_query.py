"""Command-line RAG query tool."""
import requests
import sys
from rag_utils import query_rag, QUERY_TOP_K

# Configuration
OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llama2'


def main():
    """Main entry point for command-line queries."""
    if len(sys.argv) < 2:
        print('Usage: python rag_query.py "your question here"')
        sys.exit(1)
    
    question = sys.argv[1]
    
    try:
        # Get context from RAG
        _, context_chunks, metadatas = query_rag(question, QUERY_TOP_K)
        context = '\n'.join(context_chunks) if context_chunks else "[No context found]"
        
        # Build prompt for LLM
        prompt = f"""You are an AI assistant with access to the following document excerpts:\n\n{context}\n\nUsing ONLY this information, answer the following question as accurately as possible:\n\nQuestion: {question}\nAnswer:"""
        
        # Query Ollama
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = response.json()
        answer = data.get('response', '[No answer received]')
        
        print("\n=== RAG Answer ===\n")
        print(answer)
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Failed to connect to Ollama. Make sure Ollama is running.\n{str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
