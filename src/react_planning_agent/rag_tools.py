from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.react_planning_agent.pdf_tools import read_text

ProgressCallback = Callable[[float, str], None]


@dataclass
class QdrantRagIndex:
    pages: list[str]
    client: QdrantClient
    collection_name: str
    vector_size: int
    embedding_provider: "EmbeddingProvider"
    chunks: list["_Chunk"]

    def search(
        self,
        query: str,
        top_k: int = 3,
        sections: list[str] | None = None,
    ) -> list[dict[str, str | int]]:
        if not query.strip():
            return []

        query_vector = self.embedding_provider.embed_query(query)
        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(1, top_k * 4),
        )
        results = list(getattr(query_result, "points", []))

        matches: list[dict[str, str | int]] = []
        for result in results:
            payload = result.payload or {}
            page_number = int(payload.get("page_number", 1))
            text = str(payload.get("text", ""))
            section = str(payload.get("section", "General"))
            matches.append({"page_number": page_number, "text": text, "section": section})

        if sections:
            section_filtered = [
                match for match in matches if _section_matches(str(match.get("section", "")), sections)
            ]
            if section_filtered:
                matches = section_filtered

        return matches[: max(1, top_k)]

    def read_text(self, start_page: int, end_page: int) -> str:
        return read_text(self.pages, start_page=start_page, end_page=end_page)

    def extract_sections(self, sections: list[str], max_chunks: int = 8) -> str:
        if not sections:
            return ""

        matched_chunks = [chunk for chunk in self.chunks if _section_matches(chunk.section, sections)]
        if matched_chunks:
            limited_chunks = matched_chunks[: max(1, max_chunks)]
            return "\n\n".join(
                f"Page {chunk.page_number} ({chunk.section}): {chunk.text}" for chunk in limited_chunks
            )

        fallback_query = " ".join(section.strip() for section in sections if section.strip())
        matches = self.search(fallback_query, top_k=max_chunks, sections=sections)
        return "\n\n".join(
            f"Page {match['page_number']} ({match.get('section', 'General')}): {match['text']}" for match in matches
        )


def build_rag_index(pages: list[str], on_progress: ProgressCallback | None = None) -> QdrantRagIndex:
    _notify_progress(on_progress, 0.05, "Preparing chunks")
    chunks = _chunk_pages(pages)

    _notify_progress(on_progress, 0.2, "Preparing embeddings")
    embedding_provider = EmbeddingProvider.from_env()

    _notify_progress(on_progress, 0.45, "Embedding chunks")
    vectors = embedding_provider.embed_documents([
        _chunk_embedding_text(chunk) for chunk in chunks
    ])

    vector_size = len(vectors[0]) if vectors else embedding_provider.dimension
    client = QdrantClient(location=":memory:")
    collection_name = f"pdf_chunks_{uuid.uuid4().hex}"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    _notify_progress(on_progress, 0.75, "Indexing vectors in Qdrant")
    points = [
        PointStruct(
            id=index,
            vector=vector,
            payload={
                "page_number": chunk.page_number,
                "text": chunk.text,
                "section": chunk.section,
                "section_slug": _normalize_section_name(chunk.section),
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False), start=1)
    ]
    if points:
        client.upsert(collection_name=collection_name, points=points)

    _notify_progress(on_progress, 1.0, "RAG index ready")
    return QdrantRagIndex(
        pages=pages,
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        embedding_provider=embedding_provider,
        chunks=chunks,
    )


@dataclass
class _Chunk:
    page_number: int
    text: str
    section: str


class EmbeddingProvider:
    def __init__(self, mode: str, dimension: int = 256) -> None:
        self.mode = mode
        self.dimension = dimension
        self.model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        self.client: OpenAI | None = None

        if self.mode == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if api_key:
                self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    @classmethod
    def from_env(cls) -> "EmbeddingProvider":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if api_key:
            return cls(mode="openrouter")
        return cls(mode="local")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return [[0.0] * self.dimension]
        if self.mode == "openrouter" and self.client is not None:
            return self._embed_documents_remote(texts)
        return [self._embed_local(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        if self.mode == "openrouter" and self.client is not None:
            response = self.client.embeddings.create(model=self.model, input=query)
            return list(response.data[0].embedding)
        return self._embed_local(query)

    def _embed_documents_remote(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)  # type: ignore[union-attr]
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors

    def _embed_local(self, text: str) -> list[float]:
        return _embed_local_text(text=text, dimension=self.dimension)


def _chunk_pages(
    pages: list[str],
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 120,
    similarity_threshold: float = 0.48,
) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    active_section = "General"
    for page_number, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue

        page_section = _extract_page_section_heading(text) or active_section
        active_section = page_section

        sentences = _split_sentences(text)
        if not sentences:
            continue

        current_chunk_sentences: list[str] = []
        previous_sentence_vector: list[float] | None = None

        for sentence in sentences:
            sentence_vector = _embed_local_text(sentence)
            chunk_text = " ".join(current_chunk_sentences)
            would_exceed_max_size = bool(chunk_text) and (len(chunk_text) + 1 + len(sentence) > max_chunk_chars)

            should_split_on_semantics = False
            if current_chunk_sentences and previous_sentence_vector is not None:
                similarity = _cosine_similarity(previous_sentence_vector, sentence_vector)
                should_split_on_semantics = similarity < similarity_threshold and len(chunk_text) >= min_chunk_chars

            if current_chunk_sentences and (would_exceed_max_size or should_split_on_semantics):
                chunks.append(_Chunk(page_number=page_number, text=chunk_text, section=page_section))
                current_chunk_sentences = []

            current_chunk_sentences.append(sentence)
            previous_sentence_vector = sentence_vector

        if current_chunk_sentences:
            chunks.append(
                _Chunk(page_number=page_number, text=" ".join(current_chunk_sentences), section=page_section)
            )

    if not chunks and pages:
        chunks.append(_Chunk(page_number=1, text=pages[0][:max_chunk_chars], section="General"))
    return chunks


def _split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text)
    return [candidate.strip() for candidate in candidates if candidate.strip()]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_local_text(text: str, dimension: int = 256) -> list[float]:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return [0.0] * dimension

    vector = [0.0] * dimension
    for token in tokens:
        token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        index = token_hash % dimension
        vector[index] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm > 0:
        return [value / norm for value in vector]
    return vector


def _extract_page_section_heading(page_text: str) -> str | None:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:3]:
        candidate = _clean_section_heading(line)
        if candidate is not None:
            return candidate
    return None


def _clean_section_heading(value: str) -> str | None:
    compact = re.sub(r"\s+", " ", value).strip(" :\t")
    if not compact:
        return None

    if len(compact) > 90:
        return None

    heading_pattern = r"^(?:[0-9]+(?:\.[0-9]+)*[.)]?\s+)?[A-Za-z][A-Za-z0-9 ,&\-/]{1,80}$"
    if re.match(heading_pattern, compact):
        return compact

    return None


def _normalize_section_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _section_matches(section_name: str, requested_sections: list[str]) -> bool:
    normalized_section = _normalize_section_name(section_name)
    if not normalized_section:
        return False

    for requested in requested_sections:
        normalized_requested = _normalize_section_name(requested)
        if normalized_requested and normalized_requested in normalized_section:
            return True
    return False


def _chunk_embedding_text(chunk: _Chunk) -> str:
    return f"Section: {chunk.section}\n{chunk.text}"


def _notify_progress(on_progress: ProgressCallback | None, value: float, message: str) -> None:
    if on_progress is not None:
        on_progress(value, message)
