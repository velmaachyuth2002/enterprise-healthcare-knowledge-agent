from pydantic import BaseModel

from app.services.document_index import DocumentIndex, ScoredChunk

_TOP_K = 3


class DocumentSearchInput(BaseModel):
    query: str


class DocumentSearchResult(BaseModel):
    found: bool
    chunks: list[ScoredChunk] = []


class DocumentSearchTool:
    name = "search_documents"
    description = (
        "Search internal company policy and guide documents for information "
        "relevant to a free-text question."
    )

    def __init__(self, index: DocumentIndex) -> None:
        self._index = index

    def run(self, params: DocumentSearchInput) -> DocumentSearchResult:
        chunks = self._index.search(params.query, top_k=_TOP_K)
        return DocumentSearchResult(found=bool(chunks), chunks=chunks)
