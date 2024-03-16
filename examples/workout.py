from youtube_qa.youtube_video_index import VideoIndexQueryResponse, YouTubeVideoIndex
from dotenv import load_dotenv

load_dotenv()

video_index = YouTubeVideoIndex()
video_index.build_index(
    search_term="huberman motivation",
    video_results=3,
)
response: VideoIndexQueryResponse = video_index.answer_question(
    question="what are the best researched supplements to help with exercise motivation",
)

print(response.answer)
print(response.sources)
