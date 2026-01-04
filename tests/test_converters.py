"""Unit tests for converter functions."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

from youtube_qa.converters import (
    transcript_to_video_info,
    transcripts_to_documents,
    sources_to_video_sources,
)
from youtube_qa.models import VideoInfo, VideoSource


class TestTranscriptToVideoInfo:
    """Test cases for transcript_to_video_info function."""

    @patch('youtube_qa.converters.YouTubeTranscriptApi')
    def test_transcript_to_video_info(self, mock_api):
        """Test converting transcript dict to VideoInfo."""
        # Setup mock transcript data
        mock_transcript_parts = [
            {"text": "Hello ", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
        ]
        mock_api.get_transcript.return_value = mock_transcript_parts

        transcript_dict = {
            "id": "test_video_id",
            "title": "Test Video Title",
            "url_suffix": "/watch?v=test_video_id",
            "views": "1,234 views",
            "publish_time": "2024-01-01",
            "thumbnails": ["thumb1.jpg", "thumb2.jpg"],
        }

        result = transcript_to_video_info(transcript_dict)

        assert isinstance(result, VideoInfo)
        assert result.id == "test_video_id"
        assert result.title == "Test Video Title"
        assert result.transcript == "Hello world"
        assert result.url == "/watch?v=test_video_id"
        assert result.views == 1234
        assert result.date == "2024-01-01"
        assert result.thumbnails == ["thumb1.jpg", "thumb2.jpg"]
        mock_api.get_transcript.assert_called_once_with("test_video_id")

    @patch('youtube_qa.converters.YouTubeTranscriptApi')
    def test_transcript_to_video_info_with_comma_in_views(self, mock_api):
        """Test handling views with commas."""
        mock_api.get_transcript.return_value = [{"text": "test"}]

        transcript_dict = {
            "id": "test_id",
            "title": "Test",
            "url_suffix": "/watch?v=test",
            "views": "1,234,567 views",
            "publish_time": "2024-01-01",
            "thumbnails": [],
        }

        result = transcript_to_video_info(transcript_dict)
        assert result.views == 1234567


class TestTranscriptsToDocuments:
    """Test cases for transcripts_to_documents function."""

    def test_transcripts_to_documents(self):
        """Test converting VideoInfo list to Document list."""
        transcripts = [
            VideoInfo(
                title="Video 1",
                id="id1",
                transcript="Transcript 1",
                url="url1",
                views=1000,
                date="2024-01-01",
                thumbnails=["thumb1"],
            ),
            VideoInfo(
                title="Video 2",
                id="id2",
                transcript="Transcript 2",
                url="url2",
                views=2000,
                date="2024-01-02",
                thumbnails=["thumb2"],
            ),
        ]

        result = transcripts_to_documents(transcripts)

        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)
        assert result[0].text == "Transcript 1"
        assert result[0].extra_info["title"] == "Video 1"
        assert result[0].extra_info["id"] == "id1"
        assert result[0].extra_info["url"] == "url1"
        assert result[0].extra_info["thumbnails"] == ["thumb1"]
        assert result[1].text == "Transcript 2"
        assert result[1].extra_info["title"] == "Video 2"

    def test_transcripts_to_documents_empty_list(self):
        """Test converting empty list."""
        result = transcripts_to_documents([])
        assert result == []


class TestSourcesToVideoSources:
    """Test cases for sources_to_video_sources function."""

    def test_sources_to_video_sources(self):
        """Test converting NodeWithScore list to VideoSource list."""
        node1 = TextNode(text="Content 1")
        node1.metadata = {
            "title": "Video 1",
            "id": "id1",
            "url": "url1",
            "thumbnails": ["thumb1"],
        }

        node2 = TextNode(text="Content 2")
        node2.metadata = {
            "title": "Video 2",
            "id": "id2",
            "url": "url2",
            "thumbnails": ["thumb2"],
        }

        sources = [
            NodeWithScore(node=node1, score=0.9),
            NodeWithScore(node=node2, score=0.8),
        ]

        result = sources_to_video_sources(sources)

        assert len(result) == 2
        assert all(isinstance(vs, VideoSource) for vs in result)
        assert result[0].title == "Video 1"
        assert result[0].id == "id1"
        assert result[0].url == "url1"
        assert result[0].thumbnails == ["thumb1"]
        assert result[1].title == "Video 2"
        assert result[1].id == "id2"

    def test_sources_to_video_sources_empty_list(self):
        """Test converting empty list."""
        result = sources_to_video_sources([])
        assert result == []

    def test_sources_to_video_sources_single_source(self):
        """Test converting single source."""
        node = TextNode(text="Content")
        node.metadata = {
            "title": "Single Video",
            "id": "single_id",
            "url": "single_url",
            "thumbnails": ["single_thumb"],
        }

        sources = [NodeWithScore(node=node, score=0.95)]
        result = sources_to_video_sources(sources)

        assert len(result) == 1
        assert result[0].title == "Single Video"
        assert result[0].id == "single_id"

