from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def read_text(pages: list[str], start_page: int, end_page: int) -> str:
    bounded_start = max(1, start_page)
    bounded_end = min(len(pages), end_page)
    selected_pages = pages[bounded_start - 1 : bounded_end]
    return "\n\n".join(selected_pages).strip()


def search_pdf(pages: list[str], query: str, top_k: int = 3) -> list[dict[str, str | int]]:
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    scored_pages: list[tuple[int, int, str]] = []

    for index, text in enumerate(pages, start=1):
        lower_text = text.lower()
        score = sum(lower_text.count(term) for term in terms)
        if score > 0:
            scored_pages.append((score, index, text))

    if not scored_pages and pages:
        return [{"page_number": 1, "text": pages[0][:1000]}]

    scored_pages.sort(key=lambda item: item[0], reverse=True)
    return [
        {"page_number": page_number, "text": page_text[:1000]}
        for _, page_number, page_text in scored_pages[:top_k]
    ]

