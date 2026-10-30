from src.react_planning_agent.rag_tools import _chunk_pages, build_rag_index


def test_qdrant_rag_index_retrieves_relevant_chunks() -> None:
    pages = [
        "General background and setup notes.",
        "The damages estimate is 2 million dollars according to the filing.",
        "Closing page.",
    ]

    index = build_rag_index(pages=pages)
    matches = index.search("damages estimate", top_k=1)

    assert len(matches) == 1
    assert matches[0]["page_number"] == 2


def test_qdrant_rag_index_reads_page_ranges() -> None:
    pages = [
        "First page details.",
        "Second page with damages estimate.",
        "Third page appendix.",
    ]

    index = build_rag_index(pages=pages)
    text = index.read_text(start_page=2, end_page=3)

    assert "Second page" in text
    assert "Third page" in text


def test_semantic_splitter_breaks_chunks_on_topic_shift() -> None:
    pages = [
        (
            "The merger agreement includes termination fees and indemnities. "
            "The transaction closes in Q4 pending approvals. "
            "Photosynthesis converts light into chemical energy in chloroplasts. "
            "Plants use carbon dioxide and water to produce glucose."
        )
    ]

    chunks = _chunk_pages(pages)

    assert len(chunks) >= 2


def test_qdrant_rag_index_extracts_multiple_pages_for_sections() -> None:
    pages = [
        "1. Background\nOverview of the filing and parties.",
        "2. Damages\nDamages include direct losses of 2 million dollars.",
        "2. Damages\nAdditional damages include legal costs and interest.",
    ]

    index = build_rag_index(pages=pages)
    text = index.extract_sections(["damages"], max_chunks=8)

    assert "2 million dollars" in text
    assert "legal costs" in text
