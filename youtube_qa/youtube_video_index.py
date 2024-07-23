from __future__ import annotations
from typing import NamedTuple
from llama_index.core.base.base_query_engine import BaseQueryEngine
from youtube_search import YoutubeSearch
from llama_index.core import Document, Settings, VectorStoreIndex
from youtube_qa.converters import (
    sources_to_video_sources,
    transcript_to_video_info,
    transcripts_to_documents,
)
from youtube_qa.sentence_transformer import SentenceTransformerEmbedding
from youtube_qa.models import VideoInfo, VideoSource
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
import logging


class VideoIndexQueryResponse(NamedTuple):
    sources: list[VideoSource]
    answer: str


class YouTubeVideoIndex:
    def __init__(
        self,
        llm: LLM | None = None,
        embed_model: BaseEmbedding | None = None,
        *,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo-0125",
    ) -> None:
        self._llm: LLM = llm if llm else OpenAI(model=llm_model_name)
        self._index: VectorStoreIndex | None = None
        self._embedding_model: BaseEmbedding = (
            embed_model
            if embed_model
            else SentenceTransformerEmbedding(model_name=embedding_model_name)
        )
        Settings.embed_model = self._embedding_model

    def build_index(
        self,
        search_term: str,
        chunk_size: int = 500,
        video_results: int = 5,
        show_progress: bool = True,
    ) -> None:
        """Answer the referenced question using YouTube search results as context.

        Args:
            search_term: The search term to use to find relevant videos.
            chunk_size: The chunk size to use for the index.
            video_results: The number of videos to use as context.
            show_progress: Whether to show the progress bar or not when generating embeddings.
        """
        results: list[dict] = YoutubeSearch(
            search_term,
            max_results=video_results,
        ).to_dict()  # type: ignore
        transcripts: list[VideoInfo] = []

        for result in results:
            logging.debug("Getting transcript for video '" + result["title"] + "'...")
            transcripts.append(transcript_to_video_info(result))

        documents: list[Document] = transcripts_to_documents(transcripts)
        Settings.chunk_size = chunk_size
        self._index = VectorStoreIndex.from_documents(
            documents, show_progress=show_progress
        )

    def answer_question(
        self,
        question: str,
        *,
        detailed: bool = True,
    ) -> VideoIndexQueryResponse:
        """Answer the referenced question using YouTube search results as context.

        Args:
            question: The question to answer.
            detailed: Whether to return detailed information about the answer.

        Returns:
            An answer to the question.
        """
        if self._index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if detailed:
            question += ". Provide a detailed response to fully answer the question using all information you have."

        query_engine: BaseQueryEngine = self._index.as_query_engine(llm=self._llm)
        response = query_engine.query(question)
        sources: list[VideoSource] = sources_to_video_sources(response.source_nodes)
        return VideoIndexQueryResponse(sources, str(response))

    def generate_search_query(self, question: str) -> str:
        """Get a relevant search query for the given question.

        Args:
            question: The question to get the relevant search query for.

        Returns:
            The relevant search query.
        """
        system_prompt = """Given the following user question, return a concise but targeted search query
        to find relevant YouTube videos that will answer the question. Return only the search query
        and nothing else, no prefix or explanation."""
        full_prompt: str = (
            f"{system_prompt}\n\nUser Question: {question}\n\nSearch Query:"
        )
        return self._llm.complete(full_prompt).text
