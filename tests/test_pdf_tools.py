from pypdf import PdfWriter

from src.react_planning_agent.pdf_tools import extract_pages_from_pdf, search_pdf


def test_extract_pages_from_pdf_returns_all_pages() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)

    pdf_bytes = bytearray()
    with open("/tmp/test_react_agent.pdf", "wb") as file_obj:
        writer.write(file_obj)
    with open("/tmp/test_react_agent.pdf", "rb") as file_obj:
        pdf_bytes = file_obj.read()

    pages = extract_pages_from_pdf(pdf_bytes)

    assert len(pages) == 2


def test_search_pdf_prioritizes_relevant_pages() -> None:
    pages = [
        "General background and setup notes.",
        "The damages estimate is 2 million dollars according to the filing.",
        "Closing page.",
    ]

    matches = search_pdf(pages=pages, query="damages estimate", top_k=1)

    assert len(matches) == 1
    assert matches[0]["page_number"] == 2

