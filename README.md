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
from youtube_qa.answer_question import answer_question_using_youtube

response, sources = answer_question_using_youtube(
    search_term="peter attia running endurance",
    question="how to train for endurance",
)

print(response)
```

Or to let the LLM auto-generate a relevant search query:

```python
from youtube_qa.answer_question import answer_question_using_youtube

response, sources = answer_question_using_youtube(
    question="how to train for endurance",
)

print(response)
```
