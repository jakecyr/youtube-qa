from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from youtube_qa.embedding import SentenceTransformerEmbedding
from dotenv import load_dotenv
import logging
from youtube_qa.get_relevant_search_query import get_relevant_search_query_for_question
from youtube_qa.models import VideoInfo, VideoSource

load_dotenv()


def transcript_to_video_info(transcript: dict) -> VideoInfo:
    transcript_parts: list[dict] = YouTubeTranscriptApi.get_transcript(transcript["id"])
    full_transcript: str = "".join(list(map(lambda x: x["text"], transcript_parts)))
    return VideoInfo(
        title=transcript["title"],
        id=transcript["id"],
        transcript=full_transcript,
        url=transcript["url_suffix"],
        views=int(transcript["views"].replace(" views", "").replace(",", "")),
        date=transcript["publish_time"],
        thumbnails=transcript["thumbnails"],
    )


def transcripts_to_documents(transcripts: list[VideoInfo]) -> list[Document]:
    return [
        Document(
            text=t.transcript,
            extra_info={
                "title": t.title,
                "url": t.url,
                "id": t.id,
                "thumbnails": t.thumbnails,
            },
        )
        for t in transcripts
    ]


def sources_to_video_sources(sources: list[NodeWithScore]) -> list[VideoSource]:
    return [
        VideoSource(
            title=source.metadata["title"],
            id=source.metadata["id"],
            url=source.metadata["url"],
            thumbnails=source.metadata["thumbnails"],
        )
        for source in sources
    ]


def answer_question_using_youtube(
    question: str,
    search_term: str | None = None,
    chunk_size: int = 500,
    video_results: int = 5,
) -> tuple[str, list[VideoSource]]:
    """Answer the referenced question using YouTube search results as context.

    Args:
        question: The question to answer.
        search_term: The search term to use to find relevant videos.
        chunk_size: The chunk size to use for the index.
        video_results: The number of videos to use as context.

    Returns:
        An answer to the question.
    """
    if search_term is None:
        logging.debug("Getting relevant search query for question...")
        search_term = get_relevant_search_query_for_question(question)

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
    response = index.as_query_engine().query(question)
    sources: list[VideoSource] = sources_to_video_sources(response.source_nodes)
    return str(response), sources
