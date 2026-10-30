from __future__ import annotations

import hashlib

import streamlit as st

from src.react_planning_agent.service import (
    analyze_pdf_question,
    analyze_prepared_pdf_question,
    prepare_pdf_for_qa,
)

st.set_page_config(page_title="ReAct Planning Agent", page_icon="📄", layout="wide")
st.title("📄 ReAct Planning Agent for PDFs")
st.caption("Observe → Think → Act → Reflect with max-iteration safeguards.")

if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "pending_run" not in st.session_state:
    st.session_state["pending_run"] = False
if "run_payload" not in st.session_state:
    st.session_state["run_payload"] = None
if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None
if "uploaded_pdf_hash" not in st.session_state:
    st.session_state["uploaded_pdf_hash"] = None
if "prepared_pdf" not in st.session_state:
    st.session_state["prepared_pdf"] = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_file")
question = st.text_input("Ask a question about the PDF", key="question_input")
max_iterations = st.slider("Max ReAct iterations", min_value=1, max_value=8, value=4)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()

    if st.session_state.get("uploaded_pdf_hash") != uploaded_hash:
        st.session_state["uploaded_pdf_bytes"] = uploaded_bytes
        st.session_state["uploaded_pdf_hash"] = uploaded_hash
        st.session_state["latest_result"] = None

        progress_bar = st.progress(0.0, text="Preparing RAG index...")

        def _on_progress(value: float, message: str) -> None:
            progress_bar.progress(max(0.0, min(1.0, value)), text=message)

        with st.spinner("Indexing uploaded PDF into Qdrant..."):
            prepared_pdf = prepare_pdf_for_qa(pdf_bytes=uploaded_bytes, on_progress=_on_progress)
        st.session_state["prepared_pdf"] = prepared_pdf
        progress_bar.progress(1.0, text="RAG index ready")


if st.button("Run ReAct Agent", disabled=bool(st.session_state["is_processing"])):
    pdf_bytes = st.session_state.get("uploaded_pdf_bytes")
    if not pdf_bytes:
        st.error("Please upload a PDF before running the agent.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        st.session_state["is_processing"] = True
        st.session_state["pending_run"] = True
        st.session_state["run_payload"] = {
            "pdf_bytes": pdf_bytes,
            "question": question.strip(),
            "max_iterations": max_iterations,
        }
        st.rerun()


if st.session_state["pending_run"]:
    payload = st.session_state["run_payload"]
    if payload is not None:
        with st.spinner("Running ReAct agent..."):
            prepared_pdf = st.session_state.get("prepared_pdf")
            if prepared_pdf is not None:
                result = analyze_prepared_pdf_question(
                    prepared_pdf=prepared_pdf,
                    question=payload["question"],
                    max_iterations=payload["max_iterations"],
                )
            else:
                result = analyze_pdf_question(
                    pdf_bytes=payload["pdf_bytes"],
                    question=payload["question"],
                    max_iterations=payload["max_iterations"],
                )
        st.session_state["latest_result"] = result
    st.session_state["pending_run"] = False
    st.session_state["is_processing"] = False
    st.session_state["run_payload"] = None
    st.rerun()

latest_result = st.session_state.get("latest_result")
if latest_result is not None:
    if latest_result.completed:
        st.success(latest_result.answer)
    else:
        st.warning(latest_result.answer)

    with st.expander("ReAct trace", expanded=False):
        for index, step in enumerate(latest_result.steps, start=1):
            st.markdown(f"**Step {index} Thought:** {step.thought}")
            st.markdown(f"**Action:** `{step.action.name}` with `{step.action.args}`")
            st.markdown(f"**Observation:** {step.observation}")
            st.markdown(f"**Self-critique:** {step.critique}")
            st.divider()

