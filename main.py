"""Command-line entry point for the CSV Analyst Agent.

Asks the user for a CSV file, loads it, then runs an interactive loop where they
ask questions in plain English and the agent answers using its tools. Everything
runs in the terminal: no browser, no web UI.

Run it with:
    python main.py
"""

import string
import sys

from agent import build_agent
from data_state import set_active_dataframe
from utils import extract_answer, load_file_with_retry

# The agent's name, used in the banner and on every answer it gives. Defined
# once here so it can be changed in a single place.
AGENT_NAME = "Agent Athreya"

# Words that signal the user wants to end the session. Detection is intent-based
# rather than exact-match, so "bye" and "i want to exit" both work, not only the
# literal word "exit".
EXIT_KEYWORDS = {"exit", "quit", "q", "bye", "goodbye", "stop", "end"}


def is_exit_request(text: str) -> bool:
    """Return True if a short user message is asking to end the session.

    The message must be brief and contain a clear exit word, so inputs like
    "bye", "stop", or "i want to exit" end the session, while a longer data
    question that merely happens to include such a word does not.
    """
    words = [word.strip(string.punctuation) for word in text.lower().split()]
    if not words or len(words) > 4:
        return False
    return any(word in EXIT_KEYWORDS for word in words)


def ask_agent(agent, question: str) -> str:
    """Send one question to the agent and return its answer as text.

    Any single turn can fail for reasons outside our control: a network blip, a
    rate limit, or the model occasionally producing a malformed tool call. When
    that happens the error is reported and the conversation continues, instead of
    letting one bad turn crash the whole session.
    """
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": "cli-session"}},
        )
    except Exception as error:
        return f"Sorry, that question failed ({error}). Try rephrasing it."
    return extract_answer(result)


def run_conversation(agent, file_label: str) -> None:
    """Run the interactive question-and-answer loop until the user exits."""
    print(f"\nReady — analysing '{file_label}'.")
    print("Ask a question about your data, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if is_exit_request(question):
            print(f"{AGENT_NAME}: Goodbye!")
            return

        print(f"{AGENT_NAME}: {ask_agent(agent, question)}\n")


def main() -> None:
    """Ask for a CSV, build the agent, and start the conversation."""
    print(f"{AGENT_NAME} — CSV Analyst")
    print("------------------------------")

    dataframe, file_path = load_file_with_retry(
        "Enter the path to your CSV or Excel file (or 'exit' to quit): "
    )
    set_active_dataframe(dataframe)

    try:
        agent = build_agent()
    except RuntimeError as error:
        print(f"Could not start agent: {error}")
        sys.exit(1)

    run_conversation(agent, file_path)


if __name__ == "__main__":
    main()
