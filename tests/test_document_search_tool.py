from pathlib import Path

from app.services.document_index import DocumentIndex
from app.tools.document_search_tool import DocumentSearchInput, DocumentSearchTool


def test_returns_relevant_chunks_for_a_real_question(document_index: DocumentIndex):
    tool = DocumentSearchTool(document_index)

    result = tool.run(DocumentSearchInput(query="What are the password requirements?"))

    assert result.found is True
    assert result.chunks[0].source == "hipaa_security_policy.md"


def test_returns_not_found_when_the_index_is_empty(tmp_path: Path):
    tool = DocumentSearchTool(DocumentIndex(tmp_path))

    result = tool.run(DocumentSearchInput(query="anything"))

    assert result.found is False
    assert result.chunks == []
