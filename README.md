# YouTube Question Answer

Simple experiment for question answering on YouTube videos using embeddings and
the top n YouTube search result transcripts.

The function will take a question and optionally a YouTube search query (otherwise an LLM will auto-generate one),
will compile transcripts for each video result, generate an embedding index using the transcripts and then answer the
question using the relevant embeddings.

The function will return both a string response and a list of sources that were used for the answer.

## Installation

The package can be installed from PyPI with `pip install youtube-qa`. Make sure to set your `OPENAI_API_KEY`
environment variable before using.

## Example

```python
from youtube_qa.youtube_video_index import VideoIndexQueryResponse, YouTubeVideoIndex

video_index = YouTubeVideoIndex()
video_index.build_index(
    search_term="huberman motivation",
    video_results=3,
)
response: VideoIndexQueryResponse = video_index.answer_question(
    question="what are the best researched supplements to help with exercise motivation",
)

print(response.answer) # The answer to the question.
print(response.sources) # Video links and other metadata.
```

You can also generate the search query given the question:

```python
from youtube_qa.youtube_video_index import VideoIndexQueryResponse, YouTubeVideoIndex

question = "what are the best researched supplements to help with exercise motivation"
video_index = YouTubeVideoIndex(
    # Can optionally pass in custom embedding model and LLM here.
)
search_term = video_index.generate_search_query(question)

video_index.build_index(
    search_term=search_term,
    video_results=3,
)
response: VideoIndexQueryResponse = video_index.answer_question(
    question=question,
)

print(response.answer) # The answer to the question.
print(response.sources) # Video links and other metadata.
```
