from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.react_planning_agent.react_loop import AgentAction, AgentResult, AgentStep

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_requires_pdf_and_question() -> None:
    at = AppTest.from_file(APP_PATH).run()

    at.button[0].click().run()

    assert "Please upload a PDF" in at.error[0].value


def test_app_renders_agent_answer(monkeypatch) -> None:
    def _fake_analyze_pdf_question(pdf_bytes: bytes, question: str, max_iterations: int) -> AgentResult:
        return AgentResult(
            answer="The contract was signed on 2024-02-14.",
            completed=True,
            steps=[
                AgentStep(
                    thought="Check for signing date.",
                    action=AgentAction(name="search_pdf", args={"query": "signed"}),
                    observation="Found date on page 3.",
                    critique="Enough evidence.",
                )
            ],
        )

    monkeypatch.setattr(
        "src.react_planning_agent.service.analyze_pdf_question",
        _fake_analyze_pdf_question,
    )

    at = AppTest.from_file(APP_PATH)
    at.session_state["uploaded_pdf_bytes"] = b"%PDF-1.4 fake"
    at.run()
    at.text_input(key="question_input").set_value("When was the contract signed?").run()
    at.button[0].click().run()

    assert "2024-02-14" in at.success[0].value


def test_run_button_is_disabled_when_processing() -> None:
    at = AppTest.from_file(APP_PATH)
    at.session_state["is_processing"] = True

    at.run()

    assert at.button[0].disabled is True

