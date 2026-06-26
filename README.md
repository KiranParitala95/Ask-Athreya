# Ask Athreya — AI Data Analyst Agent

[![CI](https://github.com/KiranParitala95/Ask-Athreya/actions/workflows/ci.yml/badge.svg)](https://github.com/KiranParitala95/Ask-Athreya/actions/workflows/ci.yml)

An AI agent that answers questions about any CSV or Excel file in plain
English. Ask it something like *"which column has the most nulls?"* or *"what's
the average salary by department?"* and it figures out how to get the answer
on its own — no SQL, no pandas, no clicking around a spreadsheet.

It runs entirely in your terminal.

```
You: which column has the most missing values?
Agent: The 'age' column has the most missing values — 10 nulls out of 63 rows.

You: average salary by department?
Agent: Sales leads at $103,489, followed by Engineering ($95,994),
       Finance ($94,984), Support ($94,143), and Marketing ($88,952).

You: are there duplicate rows?
Agent: Yes — the dataset contains 3 duplicate rows.
```

---

## What makes this an "agent" and not just a chatbot

A plain language model only talks. This one *acts*. Behind the scenes it runs a
loop: it reads your question, decides which tool it needs, calls that tool on the
real data, looks at what came back, and decides whether it has enough to answer
or needs another tool. It keeps going until the question is solved.

```
Your question
      |
      v
  The agent reasons: "what do I need to know?"
      |
      v
  Picks a tool  --->  tool runs real pandas on the CSV
      |                       |
      |  <--------------------+   result comes back
      v
  Enough to answer?  --- no ---> pick another tool
      |
     yes
      |
      v
  Plain-English answer
```

The model never touches your data directly. It can only act through four tools
that you control — and all the real work in those tools is ordinary pandas.

---

## The four tools

| Tool | What it does |
|---|---|
| `get_data_overview` | Shape of the dataset plus every column name and type |
| `inspect_columns` | Null counts and summary statistics for each column |
| `run_dataframe_query` | Runs a single pandas expression for any custom calculation |
| `find_data_quality_issues` | Flags duplicate rows, empty columns, constant columns |

The agent decides which of these to call, and in what order, based purely on the
question you ask.

---

## Project structure

```
ask-athreya/
|
|-- main.py              # CLI entry point + the interactive loop
|-- agent.py             # wires the language model to the tools
|-- tools.py             # the four data tools (plain pandas)
|-- data_loader.py       # loads + validates the CSV/Excel file
|-- data_state.py        # holds the active DataFrame for the session
|-- data/
|   |-- sample.csv       # messy sample data to try it on
|-- tests/               # unit tests for the loader, tools, state, exit logic
|-- eval_harness.py      # runs real questions through the live agent and grades them
|-- .env.example         # template for your API key (copy to .env)
|-- .gitignore
|-- requirements.txt     # runtime dependencies
|-- requirements-dev.txt # runtime + test dependencies
|-- README.md
```

Each file has one job. The agent decides *what* to do; the tools do the actual
data work; the loader and state modules keep the data clean and in one place.

---

## Tech stack

| What | Why |
|---|---|
| LangChain | Runs the agent loop and tool orchestration |
| Groq (Llama 3.3 70B) | Free, fast hosted model — configurable via `.env` |
| pandas + openpyxl | All the real data work behind the tools, including Excel |
| Python 3.10+ | Core language |

---

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/KiranParitala95/ask-athreya.git
cd ask-athreya
python -m venv venv

# Activate the virtual environment:
source venv/bin/activate          # macOS / Linux
source venv/Scripts/activate      # Git Bash on Windows

pip install -r requirements.txt
```

### 2. Add your Groq API key

A free key from [console.groq.com](https://console.groq.com) is all you need.
Copy the example env file and paste your key into it:

```bash
cp .env.example .env
```

Then open `.env` and set your key:

```
GROQ_API_KEY=your_key_here
```

The `.env` file is gitignored, so your key never gets committed. You can also
override the model there with `GROQ_MODEL` if you want to try a different one;
it defaults to `llama-3.3-70b-versatile`.

### 3. Run it

```bash
python main.py
```

It will ask you for the path to a CSV or Excel file. Point it at the sample to
start:

```
Enter the path to your CSV or Excel file (or 'exit' to quit): data/sample.csv
```

Then just start asking questions. Type `exit` (or `bye`, `quit`, `stop`) when
you're done.

---

## Try it on your own data

When the agent asks for a file, give it the path to any CSV or Excel file you
have:

```
Enter the path to your CSV or Excel file (or 'exit' to quit): path/to/your_data.xlsx
```

The agent inspects the columns first, so it adapts to whatever dataset you give
it — no configuration needed. If you mistype the path, it just asks again. For
Excel files with multiple sheets, only the first sheet is read.

---

## Running the tests

The project has a small unit-test suite covering the loader, the data tools, the
shared state, and the exit detection. Install the dev dependencies and run it:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests are pure logic — they do not call the model or the network, so they
run in about a second.

---

## Evaluating the agent's real answers

The pytest suite above tests code logic, not whether the agent actually answers
correctly. A function can work perfectly and the agent can still pick the wrong
tool, guess a column name, or report a wrong number. `eval_harness.py` catches
that different class of bug by sending real questions to the live agent and
checking the answers against facts already known about `data/sample.csv`.

This makes real API calls, so it costs tokens and is not run automatically:

```bash
python eval_harness.py
```

It runs 10 questions covering each tool plus an edge case (asking about a column
that does not exist, to check the agent reports that honestly instead of
inventing an answer), and prints a pass/fail summary.

---

## A note on the query tool

`run_dataframe_query` evaluates pandas expressions in a restricted namespace
(only the DataFrame and pandas are exposed, with Python builtins stripped out).
That is fine for local, single-user analysis like this. It is deliberately *not*
hardened for running expressions from untrusted users on a public server — if you
ever take that direction, swap in a proper sandbox first.

---

## Known limitations

**Groq's free tier has two separate rate limits, and both are worth knowing about
before you hit them:**

- **A daily token budget** (e.g. 100,000 tokens/day for `llama-3.3-70b-versatile`
  at the time of writing). Heavy back-to-back testing can use this up; it resets
  on a rolling 24-hour basis, not at a fixed clock time. If you see a `429` error
  mentioning "tokens per day," this is it — the message includes how long to wait.
- **A per-request token budget** (tokens per minute, or TPM). This matters more
  on larger files: a single question can need a lot of tokens if the dataset is
  large and a tool returns many rows. Smaller/cheaper models tend to have a much
  tighter TPM ceiling than larger ones, so switching to a "lighter" model to dodge
  the daily limit can ironically make this *other* limit easier to hit.

For reference, `llama-3.3-70b-versatile` was the model that worked reliably here
against a 1,000-row CSV. A model with a noticeably smaller per-request ceiling
(`openai/gpt-oss-120b`, also on Groq) failed on a single question against the
same file with `Request too large ... tokens per minute`. Two things in this
project specifically help keep individual requests small regardless of model:
`inspect_columns` only runs full statistics on numeric columns (not on
high-cardinality text columns like names), and `run_dataframe_query` previews
are capped at `MAX_PREVIEW_ROWS` in `tools.py`.

You can check your current usage at any time at
[console.groq.com/dashboard/metrics](https://console.groq.com/dashboard/metrics).
The model is configurable via `GROQ_MODEL` in `.env`, so switching is a one-line
change — no code edits needed.

---

**Kiran Paritala** — Data Engineer
[LinkedIn](https://www.linkedin.com/in/kiran-chowdary95) · [GitHub](https://github.com/KiranParitala95) · [Email](mailto:kiranparitala95@gmail.com)
