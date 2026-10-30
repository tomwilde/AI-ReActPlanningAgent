from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.react_planning_agent.pdf_tools import read_text, search_pdf


@dataclass
class AgentAction:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    thought: str
    action: AgentAction
    observation: str = ""
    critique: str = ""


@dataclass
class AgentResult:
    answer: str
    completed: bool
    steps: list[AgentStep]


class ReasoningModel(Protocol):
    def next_step(self, question: str, steps: list[AgentStep], pages: list[str]) -> AgentStep: ...

    def critique_step(self, question: str, step: AgentStep) -> str: ...

    def fallback_answer(self, question: str, steps: list[AgentStep]) -> str: ...


class DocumentTools(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 3,
        sections: list[str] | None = None,
    ) -> list[dict[str, str | int]]: ...

    def read_text(self, start_page: int, end_page: int) -> str: ...

    def extract_sections(self, sections: list[str], max_chunks: int = 8) -> str: ...


class FakeReasoningModel:
    def __init__(self, scripted_steps: list[AgentStep]) -> None:
        self.scripted_steps = scripted_steps
        self._index = 0

    def next_step(self, question: str, steps: list[AgentStep], pages: list[str]) -> AgentStep:
        if self._index >= len(self.scripted_steps):
            return AgentStep(
                thought="I should stop with a best-effort answer.",
                action=AgentAction(name="final_answer", args={"answer": "No more scripted steps."}),
            )
        step = self.scripted_steps[self._index]
        self._index += 1
        return step

    def critique_step(self, question: str, step: AgentStep) -> str:
        if step.action.name == "search_pdf" and "No matches" in step.observation:
            return "The query was weak; try a synonym."
        return "Reasoning is coherent."

    def fallback_answer(self, question: str, steps: list[AgentStep]) -> str:
        return "I couldn't reach a confident final answer within the iteration limit."


def run_react_loop(
    model: ReasoningModel,
    question: str,
    pages: list[str],
    max_iterations: int = 4,
    document_tools: DocumentTools | None = None,
) -> AgentResult:
    steps: list[AgentStep] = []

    for _ in range(max_iterations):
        step = model.next_step(question=question, steps=steps, pages=pages)
        step.observation = _execute_action(step.action, pages, document_tools=document_tools)
        step.critique = model.critique_step(question=question, step=step)
        steps.append(step)

        if step.action.name == "final_answer":
            answer = str(step.action.args.get("answer", step.observation)).strip()
            return AgentResult(answer=answer, completed=True, steps=steps)

    return AgentResult(answer=model.fallback_answer(question=question, steps=steps), completed=False, steps=steps)


def _execute_action(action: AgentAction, pages: list[str], document_tools: DocumentTools | None = None) -> str:
    action_name = action.name.strip().lower()
    tools = document_tools or _PageDocumentTools(pages=pages)

    if action_name in {"search_pdf", "search_document"}:
        query = _resolve_query_argument(action.args)
        matches = tools.search(query=query, top_k=3)
        if not matches:
            return "No matches found."
        return "\n\n".join(
            f"Page {match['page_number']}: {match['text']!s}" for match in matches
        )

    if action_name in {"read_text", "extract_text"}:
        sections = _resolve_sections_argument(action.args)
        if sections:
            section_text = tools.extract_sections(sections=sections, max_chunks=8)
            return section_text or "No text extracted."

        start_page = _to_int(action.args.get("start_page", action.args.get("page", 1)), fallback=1)
        end_page = _to_int(action.args.get("end_page", start_page), fallback=start_page)
        text = tools.read_text(start_page=start_page, end_page=end_page)
        return text or "No text extracted."

    if action_name == "request_text":
        if "start_page" in action.args or "end_page" in action.args or "page" in action.args:
            start_page = _to_int(action.args.get("start_page", action.args.get("page", 1)), fallback=1)
            end_page = _to_int(action.args.get("end_page", start_page), fallback=start_page)
            text = tools.read_text(start_page=start_page, end_page=end_page)
            return text or "No text extracted."

        query = _resolve_query_argument(action.args)
        sections = _resolve_sections_argument(action.args)
        matches = tools.search(query=query, top_k=6, sections=sections)
        if not matches:
            return "No matches found."
        return "\n\n".join(
            f"Page {match['page_number']}: {match['text']!s}" for match in matches
        )

    if action_name == "final_answer":
        return str(action.args.get("answer", ""))

    return f"Unsupported action: {action.name}"


def _resolve_query_argument(args: dict[str, Any]) -> str:
    for key in ("query", "search_query", "text", "term", "keywords"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    sections = args.get("sections")
    if isinstance(sections, list):
        section_terms = [str(item).strip() for item in sections if str(item).strip()]
        if section_terms:
            return " ".join(section_terms)

    return ""


def _resolve_sections_argument(args: dict[str, Any]) -> list[str]:
    sections = args.get("sections")
    if isinstance(sections, list):
        return [str(item).strip() for item in sections if str(item).strip()]
    if isinstance(sections, str) and sections.strip():
        return [sections.strip()]
    section = args.get("section")
    if isinstance(section, str) and section.strip():
        return [section.strip()]
    return []


def _to_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class _PageDocumentTools:
    pages: list[str]

    def search(
        self,
        query: str,
        top_k: int = 3,
        sections: list[str] | None = None,
    ) -> list[dict[str, str | int]]:
        resolved_query = query.strip()
        if sections:
            section_query = " ".join(section for section in sections if section.strip())
            if section_query:
                resolved_query = f"{resolved_query} {section_query}".strip()
        return search_pdf(pages=self.pages, query=resolved_query, top_k=top_k)

    def read_text(self, start_page: int, end_page: int) -> str:
        return read_text(pages=self.pages, start_page=start_page, end_page=end_page)

    def extract_sections(self, sections: list[str], max_chunks: int = 8) -> str:
        section_query = " ".join(section for section in sections if section.strip())
        matches = search_pdf(pages=self.pages, query=section_query, top_k=max_chunks)
        return "\n\n".join(f"Page {match['page_number']}: {match['text']!s}" for match in matches)

