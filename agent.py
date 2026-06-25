"""Build the CSV analyst agent.

This module only wires the language model together with the data tools. The
agent's job is to decide which tool to call and when; every piece of real data
work lives in tools.py. Keeping that boundary clear makes both halves easy to
read and change on their own.

Supported providers (set the relevant key in .env):
  - Groq:   GROQ_API_KEY   + GROQ_MODEL    (default: llama-3.3-70b-versatile)
  - OpenAI: OPENAI_API_KEY + OPENAI_MODEL  (default: gpt-4o-mini)
  - Gemini: GOOGLE_API_KEY + GEMINI_MODEL  (default: gemini-2.0-flash)

If more than one key is present, the first available in the order above wins.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from data_state import get_active_dataframe
from tools import (
    find_data_quality_issues,
    get_data_overview,
    inspect_columns,
    run_dataframe_query,
)

SYSTEM_PROMPT_TEMPLATE = (
    "Your name is Agent Athreya. You are a careful data analyst answering "
    "questions about a CSV or Excel dataset that is already loaded. The dataset "
    "has exactly these columns:\n"
    "{columns}\n"
    "Always use these exact column names; never invent or shorten them. Use the "
    "provided tools to read the real data before answering, and prefer "
    "run_dataframe_query when a question needs a calculation. When a query "
    "could return many rows, select only the columns you actually need to keep "
    "results small. Keep answers short and direct, and always include the "
    "concrete numbers you found.\n\n"
    "FORMATTING RULES — follow these strictly:\n"
    "- Plain text only. No markdown, no bold (**), no asterisks, no backticks.\n"
    "- For lists, use this format: [1] Item one  [2] Item two  [3] Item three\n"
    "- For key-value pairs, use: Label: value\n"
    "- For tables, align columns with spaces.\n"
    "- Never use bullet points, dashes as bullets, or numbered lists with dots.\n"
    "- Keep answers concise. One sentence for simple facts, a short list for "
    "multiple items."
)


def _build_model():
    """Return the right LLM client based on which API key is in the environment.

    Checks providers in order (Groq, OpenAI, Gemini) and uses the first whose
    key is present. Raises RuntimeError if no provider key is set.
    """
    groq_key   = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"Using Groq ({model_name})")
        return ChatGroq(model=model_name, temperature=0)

    if openai_key:
        from langchain_openai import ChatOpenAI
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI ({model_name})")
        return ChatOpenAI(model=model_name, temperature=0)

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        print(f"Using Gemini ({model_name})")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    raise RuntimeError(
        "No API key found. Add GROQ_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY "
        "to your .env file."
    )


def build_agent():
    """Create and return the configured agent.

    Loads .env with override=True so .env always wins over stale exported
    shell variables from previous terminal sessions.
    """
    load_dotenv(override=True)

    model = _build_model()

    tools = [
        get_data_overview,
        inspect_columns,
        run_dataframe_query,
        find_data_quality_issues,
    ]

    dataframe    = get_active_dataframe()
    column_lines = "\n".join(
        f"  - {name} ({dtype})" for name, dtype in dataframe.dtypes.items()
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(columns=column_lines)

    return create_agent(
        model,
        tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
