from llama_index.core.schema import BaseNode
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core import Document, Settings, VectorStoreIndex
from youtube_qa.embedding import SentenceTransformerEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

search_term = "huberman coffee"
question = "when is the best time to drink coffee in the morning?"

embed_model = SentenceTransformerEmbedding()
Settings.embed_model = embed_model
Settings.chunk_size = 500

results: list[dict] = YoutubeSearch(search_term, max_results=2).to_dict()
transcripts: list[str] = []

for result in results:
    print("Getting transcript for video '" + result["title"] + "'...")
    transcript_parts = YouTubeTranscriptApi.get_transcript(result["id"])
    full_transcript: str = "".join(list(map(lambda x: x["text"], transcript_parts)))
    transcripts.append(full_transcript)

documents: list[Document] = [Document(text=t) for t in transcripts]
parser = SentenceSplitter()
nodes: list[BaseNode] = parser.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
response = index.as_query_engine().query(question)

print("Response:", response)
