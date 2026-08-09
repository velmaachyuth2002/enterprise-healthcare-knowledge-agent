from app.services.document_index import DocumentIndex, _split_into_chunks


def test_split_into_chunks_extracts_heading_and_body():
    text = (
        "# Some Policy\n\n"
        "## First Section\n\n"
        "First body text.\n\n"
        "## Second Section\n\n"
        "Second body text.\n"
    )

    chunks = _split_into_chunks(text, source="some_policy.md")

    assert [c.heading for c in chunks] == ["First Section", "Second Section"]
    assert chunks[0].content == "First body text."
    assert chunks[1].content == "Second body text."
    assert all(c.source == "some_policy.md" for c in chunks)


def test_split_into_chunks_ignores_text_before_first_heading():
    text = "# Title\n\nSome preamble with no heading.\n\n## Only Section\n\nBody.\n"

    chunks = _split_into_chunks(text, source="doc.md")

    assert len(chunks) == 1
    assert chunks[0].heading == "Only Section"


def test_search_finds_the_right_source_document(document_index: DocumentIndex):
    results = document_index.search("What are the password requirements?", top_k=1)

    assert results[0].source == "hipaa_security_policy.md"


def test_search_finds_a_different_source_document(document_index: DocumentIndex):
    results = document_index.search("How many PTO days do employees get per year?", top_k=1)

    assert results[0].source == "employee_handbook.md"


def test_search_respects_top_k(document_index: DocumentIndex):
    results = document_index.search("onboarding", top_k=2)

    assert len(results) == 2


def test_search_results_are_scored_and_sorted_descending(document_index: DocumentIndex):
    results = document_index.search("HIPAA training requirements", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
