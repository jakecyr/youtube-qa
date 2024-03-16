from __future__ import annotations
from typing import Any, cast
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.core.bridge.pydantic import PrivateAttr


class SentenceTransformerEmbedding(BaseEmbedding):
    _model = PrivateAttr()

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = SentenceTransformer(model_name)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        embeddings = self._model.encode(query).tolist()
        return cast(list[float], embeddings)

    def _get_text_embedding(self, text: str) -> list[float]:
        embeddings = self._model.encode([text])
        return cast(list[float], embeddings.tolist())

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts)
        return cast(list[list[float]], embeddings.tolist())
