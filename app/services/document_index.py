import re
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2: small, fast, and the de-facto default sentence-transformers
# model - 384-dim embeddings, runs on CPU in well under a second per document
# at this corpus size. Good enough for a first slice; only worth revisiting if
# retrieval quality actually falls short once the corpus grows.
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "documents"

_HEADING_PATTERN = re.compile(r"^## ", flags=re.MULTILINE)


class Chunk(BaseModel):
    source: str
    heading: str
    content: str


class ScoredChunk(Chunk):
    score: float


def _split_into_chunks(text: str, source: str) -> list[Chunk]:
    # Splits on level-2 (`## `) headings. The text before the first one is
    # just the `# Title` line with no body of its own, so it's discarded
    # rather than kept as an empty/meaningless chunk.
    sections = _HEADING_PATTERN.split(text)[1:]
    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        chunks.append(Chunk(source=source, heading=heading.strip(), content=body.strip()))
    return chunks


class DocumentIndex:
    """Embeds and indexes markdown documents for semantic search.

    Rebuilt fresh on every construction rather than persisted to disk -
    at this corpus size that costs well under a second, and it avoids an
    on-disk index silently drifting out of sync with the documents.
    """

    def __init__(self, documents_dir: Path) -> None:
        self._model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        self._client = QdrantClient(":memory:")

        chunks = [
            chunk
            for path in sorted(documents_dir.glob("*.md"))
            for chunk in _split_into_chunks(path.read_text(), source=path.name)
        ]

        self._client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self._model.get_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        if chunks:
            vectors = self._model.encode([chunk.content for chunk in chunks])
            self._client.upsert(
                collection_name=_COLLECTION_NAME,
                points=[
                    PointStruct(id=i, vector=vector.tolist(), payload=chunk.model_dump())
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        query_vector = self._model.encode(query).tolist()
        hits = self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
        ).points
        return [ScoredChunk(**hit.payload, score=hit.score) for hit in hits]
