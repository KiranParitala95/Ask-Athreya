"""Shared utilities used by both main.py and eval_harness.py."""

import sys

from data_loader import load_data_file


def extract_answer(agent_result: dict) -> str:
    """Pull the agent's final text reply out of its response object.

    Groq returns a plain string; OpenAI returns a list of content blocks.
    This normalises both into clean text.
    """
    messages = agent_result.get("messages", [])
    if not messages:
        return "(the agent returned no response)"

    content = messages[-1].content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def load_file_with_retry(prompt: str) -> tuple:
    """Ask the user for a CSV or Excel path and keep asking until one loads.

    Args:
        prompt: The input prompt shown to the user.

    Returns:
        A tuple of (DataFrame, file_path).
    """
    while True:
        path = input(prompt).strip()

        if path.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            sys.exit(0)

        if not path:
            continue

        try:
            return load_data_file(path), path
        except (FileNotFoundError, ValueError) as error:
            print(f"Could not load that file: {error}\nPlease try again.\n")
