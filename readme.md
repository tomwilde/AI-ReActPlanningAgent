# ReAct Planning Agent (Streamlit + PDF)

This app runs a ReAct-style loop (`Thought -> Action -> Observation -> Reflection`) over an uploaded PDF.

## Features

- ReAct loop with explicit max iteration limits.
- Self-critique on every step.
- Graceful degradation when iteration limits are reached.
- OpenRouter support using `OPENROUTER_API_KEY` and `openai/gpt-4.1-mini`.
- Upload-time RAG indexing with Qdrant and visual indexing progress.

## Environment

- `OPENROUTER_API_KEY` enables live LLM + embedding calls through OpenRouter.
- `OPENROUTER_MODEL` defaults to `openai/gpt-4.1-mini`.
- `OPENROUTER_EMBEDDING_MODEL` defaults to `openai/text-embedding-3-small`.

## Local development

```bash
uv sync
uv run streamlit run app.py
```

## Run with Docker Compose

Only one profile should run at a time (`app` or `debug`), so both can use port `8501`.

### App mode (default runtime profile)

```bash
make app
```

### Debug mode (VS Code attach + live edits)

```bash
make debug
```

Then in VS Code, run `Attach ReAct Debug Container` from `.vscode/launch.json`.

Stop either mode with:

```bash
make down
```

## Quality checks

```bash
make test
make lint
make qa
```

