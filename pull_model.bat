@echo off
REM Helper script to pull llama2 model into the Ollama container
REM Run this after docker-compose up

echo Pulling llama2 model into Ollama container...
docker exec rag_ollama ollama pull llama2
echo Model pulled successfully!
echo You can now use the RAG server.
pause

