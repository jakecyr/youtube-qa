"""Unit tests for YouTubeVideoIndex class."""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from youtube_qa.youtube_video_index import YouTubeVideoIndex, VideoIndexQueryResponse
from youtube_qa.models import VideoSource
from youtube_qa.sentence_transformer import SentenceTransformerEmbedding


class TestYouTubeVideoIndex:
    """Test cases for YouTubeVideoIndex."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        index = YouTubeVideoIndex()
        assert index._llm is not None
        assert index._embedding_model is not None
        assert index._index is None
        assert isinstance(index._embedding_model, SentenceTransformerEmbedding)

    def test_init_with_custom_llm(self):
        """Test initialization with custom LLM."""
        mock_llm = Mock(spec=OpenAI)
        index = YouTubeVideoIndex(llm=mock_llm)
        assert index._llm is mock_llm

    def test_init_with_custom_embed_model(self):
        """Test initialization with custom embedding model."""
        mock_embed = Mock()
        index = YouTubeVideoIndex(embed_model=mock_embed)
        assert index._embedding_model is mock_embed

    @patch('youtube_qa.youtube_video_index.SentenceTransformerEmbedding')
    def test_init_with_model_names(self, mock_embedding_class):
        """Test initialization with custom model names."""
        mock_embedding = Mock()
        mock_embedding_class.return_value = mock_embedding
        
        index = YouTubeVideoIndex(
            embedding_model_name="test-embedding",
            llm_model_name="test-llm"
        )
        assert index._llm is not None
        assert index._embedding_model is not None
        mock_embedding_class.assert_called_once_with(model_name="test-embedding")

    @patch('youtube_qa.youtube_video_index.YoutubeSearch')
    @patch('youtube_qa.youtube_video_index.transcript_to_video_info')
    @patch('youtube_qa.youtube_video_index.transcripts_to_documents')
    @patch('youtube_qa.youtube_video_index.VectorStoreIndex')
    def test_build_index(
        self,
        mock_vector_index,
        mock_transcripts_to_docs,
        mock_transcript_to_info,
        mock_youtube_search
    ):
        """Test build_index method."""
        # Setup mocks
        mock_search_instance = Mock()
        mock_search_instance.to_dict.return_value = [
            {"id": "video1", "title": "Test Video 1"},
            {"id": "video2", "title": "Test Video 2"}
        ]
        mock_youtube_search.return_value = mock_search_instance

        mock_video_info = Mock()
        mock_transcript_to_info.return_value = mock_video_info

        mock_doc = Document(text="test transcript")
        mock_transcripts_to_docs.return_value = [mock_doc]

        mock_index_instance = Mock()
        mock_vector_index.from_documents.return_value = mock_index_instance

        # Test
        index = YouTubeVideoIndex()
        index.build_index("test search", chunk_size=500, video_results=5, show_progress=False)

        # Assertions
        mock_youtube_search.assert_called_once_with("test search", max_results=5)
        assert mock_transcript_to_info.call_count == 2
        mock_transcripts_to_docs.assert_called_once()
        mock_vector_index.from_documents.assert_called_once()
        assert index._index is not None

    @patch('youtube_qa.youtube_video_index.YoutubeSearch')
    @patch('youtube_qa.youtube_video_index.transcript_to_video_info')
    @patch('youtube_qa.youtube_video_index.transcripts_to_documents')
    @patch('youtube_qa.youtube_video_index.VectorStoreIndex')
    def test_build_index_with_custom_params(
        self,
        mock_vector_index,
        mock_transcripts_to_docs,
        mock_transcript_to_info,
        mock_youtube_search
    ):
        """Test build_index with custom parameters."""
        mock_search_instance = Mock()
        mock_search_instance.to_dict.return_value = [
            {"id": "video1", "title": "Test Video 1"}
        ]
        mock_youtube_search.return_value = mock_search_instance

        mock_transcript_to_info.return_value = Mock()
        mock_transcripts_to_docs.return_value = [Document(text="test")]
        mock_vector_index.from_documents.return_value = Mock()

        index = YouTubeVideoIndex()
        index.build_index(
            "custom search",
            chunk_size=1000,
            video_results=10,
            show_progress=True
        )

        mock_youtube_search.assert_called_once_with("custom search", max_results=10)
        mock_vector_index.from_documents.assert_called_once()

    def test_answer_question_without_index(self):
        """Test answer_question raises error when index not built."""
        index = YouTubeVideoIndex()
        with pytest.raises(ValueError, match="Index not built"):
            index.answer_question("test question")

    @patch('youtube_qa.youtube_video_index.sources_to_video_sources')
    def test_answer_question_detailed(self, mock_sources_to_video):
        """Test answer_question with detailed=True."""
        # Setup
        index = YouTubeVideoIndex()
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.source_nodes = [Mock()]
        mock_response.__str__ = Mock(return_value="Test answer")
        mock_query_engine.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_query_engine
        index._index = mock_index

        mock_sources_to_video.return_value = [
            VideoSource(title="Video 1", id="id1", url="url1", thumbnails=[])
        ]

        # Test
        result = index.answer_question("test question", detailed=True)

        # Assertions
        assert isinstance(result, VideoIndexQueryResponse)
        assert result.answer == "Test answer"
        mock_query_engine.query.assert_called_once()
        call_args = mock_query_engine.query.call_args[0][0]
        assert "detailed response" in call_args.lower()
        mock_sources_to_video.assert_called_once()

    @patch('youtube_qa.youtube_video_index.sources_to_video_sources')
    def test_answer_question_not_detailed(self, mock_sources_to_video):
        """Test answer_question with detailed=False."""
        index = YouTubeVideoIndex()
        mock_index = Mock()
        mock_query_engine = Mock()
        mock_response = Mock()
        mock_response.source_nodes = []
        mock_response.__str__ = Mock(return_value="Simple answer")
        mock_query_engine.query.return_value = mock_response
        mock_index.as_query_engine.return_value = mock_query_engine
        index._index = mock_index

        mock_sources_to_video.return_value = []

        result = index.answer_question("test question", detailed=False)

        assert isinstance(result, VideoIndexQueryResponse)
        assert result.answer == "Simple answer"
        call_args = mock_query_engine.query.call_args[0][0]
        assert "detailed response" not in call_args.lower()

    @patch('youtube_qa.youtube_video_index.OpenAI')
    def test_generate_search_query(self, mock_openai):
        """Test generate_search_query method."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "optimized search query"
        mock_llm.complete.return_value = mock_response
        mock_openai.return_value = mock_llm

        index = YouTubeVideoIndex(llm=mock_llm)
        result = index.generate_search_query("What is machine learning?")

        assert result == "optimized search query"
        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args[0][0]
        assert "What is machine learning?" in call_args
        assert "search query" in call_args.lower()

