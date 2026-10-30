import pytest

from src.react_planning_agent.react_loop import (
    AgentAction,
    AgentStep,
    FakeReasoningModel,
    run_react_loop,
)


def test_react_loop_returns_final_answer_when_model_finishes() -> None:
    model = FakeReasoningModel(
        scripted_steps=[
            AgentStep(
                thought="I should locate damages references.",
                action=AgentAction(name="search_pdf", args={"query": "damages"}),
            ),
            AgentStep(
                thought="I now have enough evidence.",
                action=AgentAction(
                    name="final_answer",
                    args={"answer": "The filing states damages are 2 million dollars."},
                ),
            ),
        ]
    )

    result = run_react_loop(
        model=model,
        question="What damages are claimed?",
        pages=["No amount here.", "Damages are 2 million dollars."],
        max_iterations=4,
    )

    assert result.completed is True
    assert "2 million" in result.answer


def test_react_loop_gracefully_degrades_on_iteration_limit() -> None:
    model = FakeReasoningModel(
        scripted_steps=[
            AgentStep(
                thought="Need more context.",
                action=AgentAction(name="search_pdf", args={"query": "timeline"}),
            ),
            AgentStep(
                thought="Need still more context.",
                action=AgentAction(name="search_pdf", args={"query": "timeline"}),
            ),
            AgentStep(
                thought="Again.",
                action=AgentAction(name="search_pdf", args={"query": "timeline"}),
            ),
        ]
    )

    result = run_react_loop(
        model=model,
        question="What is the timeline?",
        pages=["Timeline item one.", "Timeline item two."],
        max_iterations=2,
    )

    assert result.completed is False
    assert "couldn't reach a confident final answer" in result.answer
    assert len(result.steps) == 2


@pytest.mark.parametrize(
    ("action_name", "action_args", "expected_fragment"),
    [
        ("search_document", {"query": "damages"}, "Damages are 2 million dollars."),
        ("extract_text", {"start_page": 2, "end_page": 2}, "Damages are 2 million dollars."),
        ("request_text", {"query": "damages"}, "Damages are 2 million dollars."),
    ],
)
def test_react_loop_supports_additional_action_aliases(
    action_name: str,
    action_args: dict[str, object],
    expected_fragment: str,
) -> None:
    model = FakeReasoningModel(
        scripted_steps=[
            AgentStep(
                thought="I should gather evidence.",
                action=AgentAction(name=action_name, args=action_args),
            ),
        ]
    )

    result = run_react_loop(
        model=model,
        question="What damages are claimed?",
        pages=["No amount here.", "Damages are 2 million dollars."],
        max_iterations=1,
    )

    assert "Unsupported action" not in result.steps[0].observation
    assert expected_fragment in result.steps[0].observation


def test_request_text_sections_is_no_longer_supported() -> None:
    model = FakeReasoningModel(
        scripted_steps=[
            AgentStep(
                thought="I should use the old sections tool.",
                action=AgentAction(name="request_text_sections", args={"sections": ["damages"]}),
            ),
        ]
    )

    result = run_react_loop(
        model=model,
        question="Summarize damages.",
        pages=["2. Damages. Damages include direct losses of 2 million dollars."],
        max_iterations=1,
    )

    assert result.steps[0].observation == "Unsupported action: request_text_sections"


def test_extract_text_by_sections_returns_multi_page_context() -> None:
    model = FakeReasoningModel(
        scripted_steps=[
            AgentStep(
                thought="I should extract the damages section.",
                action=AgentAction(name="extract_text", args={"sections": ["damages"]}),
            ),
        ]
    )

    result = run_react_loop(
        model=model,
        question="Summarize damages.",
        pages=[
            "1. Background. Introductory content only.",
            "2. Damages. Damages include direct losses of 2 million dollars.",
            "2. Damages. Additional damages include legal costs and interest.",
        ],
        max_iterations=1,
    )

    observation = result.steps[0].observation
    assert "2 million dollars" in observation
    assert "legal costs" in observation

