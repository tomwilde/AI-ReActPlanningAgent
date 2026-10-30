from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.react_planning_agent.llm_clients import build_reasoning_model
from src.react_planning_agent.pdf_tools import extract_pages_from_pdf
from src.react_planning_agent.react_loop import AgentResult, run_react_loop
from src.react_planning_agent.rag_tools import QdrantRagIndex, build_rag_index

ProgressCallback = Callable[[float, str], None]


@dataclass
class PreparedPdf:
    pages: list[str]
    rag_index: QdrantRagIndex


def prepare_pdf_for_qa(pdf_bytes: bytes, on_progress: ProgressCallback | None = None) -> PreparedPdf:
    if on_progress is not None:
        on_progress(0.05, "Extracting PDF text")

    pages = extract_pages_from_pdf(pdf_bytes)

    if on_progress is not None:
        on_progress(0.15, "Building retrieval index")
    rag_index = build_rag_index(pages=pages, on_progress=on_progress)
    return PreparedPdf(pages=pages, rag_index=rag_index)


def analyze_prepared_pdf_question(prepared_pdf: PreparedPdf, question: str, max_iterations: int) -> AgentResult:
    model = build_reasoning_model()
    return run_react_loop(
        model=model,
        question=question,
        pages=prepared_pdf.pages,
        max_iterations=max_iterations,
        document_tools=prepared_pdf.rag_index,
    )


def analyze_pdf_question(pdf_bytes: bytes, question: str, max_iterations: int) -> AgentResult:
    prepared_pdf = prepare_pdf_for_qa(pdf_bytes=pdf_bytes)
    return analyze_prepared_pdf_question(
        prepared_pdf=prepared_pdf,
        question=question,
        max_iterations=max_iterations,
    )

