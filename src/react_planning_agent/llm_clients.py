from __future__ import annotations

import json
import os
from typing import Any

from atomic_agents.base.base_io_schema import BaseIOSchema
from openai import OpenAI
from pydantic import Field, model_validator

from src.react_planning_agent.react_loop import AgentAction, AgentStep


class ReActStepSchema(BaseIOSchema):
    """Structured ReAct next-step payload emitted by the model."""

    thought: str = Field(description="Reasoning for the next action")
    action: str = Field(
        description=(
            "Action to execute: search_pdf/search_document, "
            "read_text/extract_text/request_text, final_answer"
        )
    )
    action_input: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_action_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        action = str(data.get("action", "")).strip().lower()
        action_input = data.get("action_input")
        if isinstance(action_input, dict) or action_input is None:
            return data

        if isinstance(action_input, str):
            key = "answer" if action == "final_answer" else "query"
            normalized = data.copy()
            normalized["action_input"] = {key: action_input}
            return normalized

        if isinstance(action_input, list):
            normalized = data.copy()
            if action in {"read_text", "extract_text", "request_text"}:
                normalized["action_input"] = {"sections": action_input}
            elif action in {"search_pdf", "search_document"}:
                normalized["action_input"] = {"query": " ".join(str(item) for item in action_input)}
            elif action == "final_answer":
                normalized["action_input"] = {"answer": " ".join(str(item) for item in action_input)}
            else:
                normalized["action_input"] = {"items": action_input}
            return normalized

        return data


class OpenRouterAtomicModel:
    def __init__(self, api_key: str, model: str = "openai/gpt-4.1-mini") -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def next_step(self, question: str, steps: list[AgentStep], pages: list[str]) -> AgentStep:
        compact_history = _format_history(steps)
        preview = "\n\n".join(f"Page {index + 1}: {text[:400]}" for index, text in enumerate(pages[:5]))
        prompt = (
            "You are a ReAct planning agent. Return JSON with keys "
            "thought, action, action_input. Keep thoughts concise. "
            "Supported actions: search_pdf, search_document, read_text, extract_text, "
            "request_text, final_answer. "
            f"Question: {question}\n\nHistory:\n{compact_history}\n\nPDF Preview:\n{preview}"
        )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
        )
        payload = _extract_output_text(response)
        decoded = json.loads(payload)
        schema = ReActStepSchema.model_validate(decoded)
        return AgentStep(thought=schema.thought, action=AgentAction(name=schema.action, args=schema.action_input))

    def critique_step(self, question: str, step: AgentStep) -> str:
        prompt = (
            "Critique this ReAct step in one sentence. Focus on evidence sufficiency and next-action quality. "
            f"Question: {question}\nThought: {step.thought}\nAction: {step.action.name}\n"
            f"Observation: {step.observation}"
        )
        response = self.client.responses.create(model=self.model, input=prompt, temperature=0)
        return _extract_output_text(response).strip()

    def fallback_answer(self, question: str, steps: list[AgentStep]) -> str:
        if not steps:
            return "I couldn't reach a confident final answer within the iteration limit."
        last_observation = steps[-1].observation
        return (
            "I couldn't reach a confident final answer within the iteration limit. "
            f"Best evidence gathered: {last_observation[:700]}"
        )


class LocalHeuristicModel:
    def next_step(self, question: str, steps: list[AgentStep], pages: list[str]) -> AgentStep:
        if steps:
            return AgentStep(
                thought="I should provide a concise best-effort response from current evidence.",
                action=AgentAction(
                    name="final_answer",
                    args={"answer": "I cannot use the live model right now, but relevant excerpts are in the observations."},
                ),
            )
        return AgentStep(
            thought="I should search the document for the key terms in the question.",
            action=AgentAction(name="search_pdf", args={"query": question}),
        )

    def critique_step(self, question: str, step: AgentStep) -> str:
        return "Heuristic mode enabled; result quality may be lower without live LLM reasoning."

    def fallback_answer(self, question: str, steps: list[AgentStep]) -> str:
        if steps:
            return (
                "I couldn't reach a confident final answer within the iteration limit. "
                f"Best evidence gathered: {steps[-1].observation[:700]}"
            )
        return "I couldn't reach a confident final answer within the iteration limit."


def build_reasoning_model() -> OpenRouterAtomicModel | LocalHeuristicModel:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    if api_key:
        return OpenRouterAtomicModel(api_key=api_key, model=model)
    return LocalHeuristicModel()


def _format_history(steps: list[AgentStep]) -> str:
    if not steps:
        return "No prior steps."
    return "\n".join(
        f"{idx + 1}. Thought={step.thought} | Action={step.action.name} | Observation={step.observation[:200]}"
        for idx, step in enumerate(steps)
    )


def _extract_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", "")
    if output_text:
        return str(output_text)

    output = getattr(response, "output", [])
    for item in output:
        content = getattr(item, "content", [])
        for part in content:
            text = getattr(part, "text", None)
            if text:
                return str(text)
    return "{}"

