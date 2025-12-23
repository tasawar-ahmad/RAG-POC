# Docker Setup for RAG Application

This guide explains how to run the RAG application using Docker and Docker Compose.

## Quick Start

1. **Build and start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Pull required Ollama models** (first time only):
   ```bash
   docker exec rag-ollama ollama pull llama3.2
   docker exec rag-ollama ollama pull mxbai-embed-large
   ```

3. **Check service health:**
   ```bash
   curl http://localhost:8000/health
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f rag-api
   ```

## Services

### `ollama`
- Runs the Ollama service for local LLM and embeddings
- Exposed on port `11434`
- Models are persisted in the `ollama_data` volume

### `rag-api`
- FastAPI server for the RAG application
- Exposed on port `8000`
- Depends on `ollama` service being healthy
- Automatically builds vectorstore on first startup

## Environment Variables

You can customize the application via environment variables in `docker-compose.yml`:

- `LLM_MODEL_NAME`: LLM model for text generation (default: `llama3.2`)
- `EMBEDDING_MODEL_NAME`: Embedding model (default: `mxbai-embed-large`)
- `CHUNK_SIZE`: Document chunk size (default: `1000`)
- `CHUNK_OVERLAP`: Chunk overlap (default: `200`)
- `DOCUMENTS_DIR`: Directory for source PDFs (default: `data/raw`)
- `VECTORSTORE_DIR`: Directory for vectorstore (default: `data/vectorstores`)

## Data Persistence

- **PDF Documents**: Mounted from `./data/raw` (host) to `/app/data/raw` (container)
- **Vectorstore**: Persisted in Docker volume `vectorstore_data`
- **Ollama Models**: Persisted in Docker volume `ollama_data`

## API Endpoints

Once running, access:

- Health check: `http://localhost:8000/health`
- Query endpoint: `POST http://localhost:8000/query`
  - JSON: `{"query": "your question"}`
  - Form: `text=your question&response_url=<slack_url>`

## Troubleshooting

### Ollama models not found
```bash
# Check available models
docker exec rag-ollama ollama list

# Pull missing models
docker exec rag-ollama ollama pull <model-name>
```

### Rebuild vectorstore
```bash
# Force rebuild on next query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "recreate_store": true}'
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-api
docker-compose logs -f ollama
```

### Restart services
```bash
docker-compose restart rag-api
```