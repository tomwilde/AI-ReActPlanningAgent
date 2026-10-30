from src.react_planning_agent.llm_clients import OpenRouterAtomicModel, ReActStepSchema, build_reasoning_model


def test_build_reasoning_model_uses_gpt_4_1_mini_as_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    model = build_reasoning_model()

    assert isinstance(model, OpenRouterAtomicModel)
    assert model.model == "openai/gpt-4.1-mini"


def test_build_reasoning_model_honors_openrouter_model_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1")

    model = build_reasoning_model()

    assert isinstance(model, OpenRouterAtomicModel)
    assert model.model == "openai/gpt-4.1"


def test_react_step_schema_coerces_string_action_input_for_final_answer() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I can now answer.",
            "action": "final_answer",
            "action_input": "Extract defendants and claims in the PDF bundle.",
        }
    )

    assert step.action_input == {"answer": "Extract defendants and claims in the PDF bundle."}


def test_react_step_schema_coerces_string_action_input_for_search_query() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I should search for relevant terms.",
            "action": "search_pdf",
            "action_input": "defendant claims",
        }
    )

    assert step.action_input == {"query": "defendant claims"}


def test_react_step_schema_allows_list_values_in_action_input() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I should read the requested sections.",
            "action": "read_text",
            "action_input": {"sections": ["B", "E"]},
        }
    )

    assert step.action_input == {"sections": ["B", "E"]}


def test_react_step_schema_allows_nested_object_values_in_action_input() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I should fetch a structured excerpt.",
            "action": "read_text",
            "action_input": {
                "selection": {
                    "sections": ["B", "E"],
                    "pages": {"start": 3, "end": 7},
                }
            },
        }
    )

    assert step.action_input == {
        "selection": {
            "sections": ["B", "E"],
            "pages": {"start": 3, "end": 7},
        }
    }


def test_react_step_schema_allows_list_of_objects_in_action_input() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I should issue compound lookups.",
            "action": "search_pdf",
            "action_input": {
                "queries": [
                    {"term": "defendant", "weight": 2},
                    {"term": "damages", "weight": 1},
                ]
            },
        }
    )

    assert step.action_input == {
        "queries": [
            {"term": "defendant", "weight": 2},
            {"term": "damages", "weight": 1},
        ]
    }


def test_react_step_schema_coerces_top_level_list_action_input() -> None:
    step = ReActStepSchema.model_validate(
        {
            "thought": "I should read sections B and E.",
            "action": "read_text",
            "action_input": ["B", "E"],
        }
    )

    assert step.action_input == {"sections": ["B", "E"]}
