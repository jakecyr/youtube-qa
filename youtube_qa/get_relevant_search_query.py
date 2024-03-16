from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.openai import OpenAI


def get_relevant_search_query_for_question(
    question: str,
    llm: OpenAI,
    api_key: str | None = None,
) -> str:
    """Get a relevant search query for the given question.

    Args:
        question: The question to get the relevant search query for.
        llm: The LLM instance to use.
        api_key: The OpenAI API key.

    Returns:
        The relevant search query.
    """
    system_prompt = """Given the following user question, return a concise but targeted search query
    to find relevant YouTube videos that will answer the question. Return only the search query
    and nothing else, no prefix or explanation."""
    full_prompt: str = f"{system_prompt}\n\nUser Question: {question}\n\nSearch Query:"
    completion: CompletionResponse = llm.complete(full_prompt)
    return completion.text
