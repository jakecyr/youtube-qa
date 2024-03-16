import os
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_relevant_search_query_for_question(
    question: str,
    api_key: str | None = None,
    model: str = "gpt-3.5-turbo-0125",
) -> str:
    """Get a relevant search query for the given question.

    Args:
        question: The question to get the relevant search query for.
        api_key: The OpenAI API key.
        model: The OpenAI model to use.

    Returns:
        The relevant search query.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(model=model, api_key=api_key, temperature=0)
    system_prompt = """Given the following user question, return a concise but targeted search query
    to find relevant YouTube videos that will answer the question. Return only the search query
    and nothing else, no prefix or explanation."""
    full_prompt: str = f"{system_prompt}\n\nUser Question: {question}\n\nSearch Query:"
    completion: CompletionResponse = llm.complete(full_prompt)
    return completion.text


if __name__ == "__main__":
    query: str = get_relevant_search_query_for_question("how to train for endurance")
    print(f"Query: {query}")
