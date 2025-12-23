FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION} \
    && ln -s ${POETRY_VENV}/bin/poetry /usr/local/bin/poetry \
    && poetry --version

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry: Don't create virtual env, install dependencies
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/opt/.cache

# Install dependencies
RUN poetry install --no-dev --no-root

# Copy application code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/raw data/vectorstores

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://ollama:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

