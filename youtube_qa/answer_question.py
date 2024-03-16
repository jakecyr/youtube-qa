from __future__ import annotations
from youtube_search import YoutubeSearch
from llama_index.core import Document, Settings, VectorStoreIndex
from youtube_qa.converters import (
    sources_to_video_sources,
    transcript_to_video_info,
    transcripts_to_documents,
)
from youtube_qa.embedding import SentenceTransformerEmbedding
import logging
from youtube_qa.get_relevant_search_query import get_relevant_search_query_for_question
from youtube_qa.models import VideoInfo, VideoSource
from llama_index.llms.openai import OpenAI


def answer_question_using_youtube(
    question: str,
    search_term: str | None = None,
    chunk_size: int = 500,
    video_results: int = 5,
    llm_model: str = "gpt-3.5-turbo-0125",
) -> tuple[str, list[VideoSource]]:
    """Answer the referenced question using YouTube search results as context.

    Args:
        question: The question to answer.
        search_term: The search term to use to find relevant videos.
        chunk_size: The chunk size to use for the index.
        video_results: The number of videos to use as context.
        llm_model: The OpenAI model to use.

    Returns:
        An answer to the question.
    """
    llm = OpenAI(model=llm_model, temperature=0)

    if search_term is None:
        logging.debug("Getting relevant search query for question...")
        search_term = get_relevant_search_query_for_question(question, llm)

    results: list[dict] = YoutubeSearch(
        search_term,
        max_results=video_results,
    ).to_dict()  # type: ignore
    transcripts: list[VideoInfo] = []

    for result in results:
        print("Getting transcript for video '" + result["title"] + "'...")
        transcripts.append(transcript_to_video_info(result))

    documents: list[Document] = transcripts_to_documents(transcripts)
    embed_model = SentenceTransformerEmbedding()
    Settings.embed_model = embed_model
    Settings.chunk_size = chunk_size
    index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine(llm=llm).query(question)
    sources: list[VideoSource] = sources_to_video_sources(response.source_nodes)
    return str(response), sources
