from __future__ import annotations
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore
from youtube_qa.models import VideoInfo, VideoSource


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
