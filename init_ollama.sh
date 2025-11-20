#!/bin/bash
# Wait for Ollama to be ready and pull the llama2 model

echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama..."
    sleep 2
done

echo "Ollama is ready! Pulling llama2 model..."
ollama pull llama2

echo "Model pulled successfully!"

