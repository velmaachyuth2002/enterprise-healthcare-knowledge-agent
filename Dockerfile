FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copied and synced before the app source, so this layer - installing
# sentence-transformers/torch/qdrant-client/etc, the slow part - stays
# cached across rebuilds that only change application code.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ ./app/
COPY documents/ ./documents/
COPY scripts/ ./scripts/
COPY web/ ./web/

# Pre-download the embedding model at build time rather than on first
# request, so the container never needs network access just to start
# serving traffic.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
