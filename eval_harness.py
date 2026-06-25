"""
Eval Harness for Agent Athreya

Loads any CSV or Excel file, generates test questions from the real data,
asks the agent those questions, and checks if the answers are correct.

Run it manually:
    python eval_harness.py
"""

import sys
import time

from agent import build_agent
from data_state import set_active_dataframe
from utils import extract_answer, load_file_with_retry


def get_null_info(df):
    """Return the column name and count for the column with the most nulls."""
    null_counts = df.isnull().sum()
    if null_counts.max() == 0:
        return None, 0
    col = null_counts.idxmax()
    return col, int(null_counts[col])


def build_eval_cases(df):
    """Build 15-20 test questions and expected answers from the real data in df."""
    cases = []
    row_count   = len(df)
    col_count   = len(df.columns)
    dup_count   = int(df.duplicated().sum())
    total_nulls = int(df.isnull().sum().sum())
    num_cols    = [c for c in df.select_dtypes(include="number").columns
                   if "id" not in c.lower()]
    text_cols   = [c for c in df.select_dtypes(include="object").columns
                   if df[c].nunique() / len(df) < 0.1]

    # --- Always included ---

    cases.append({
        "question": "How many rows are in this dataset?",
        "expects":  [str(row_count), f"{row_count:,}"],
        "label":    "row count",
    })

    cases.append({
        "question": "How many columns does this dataset have?",
        "expects":  [str(col_count)],
        "label":    "column count",
    })

    cases.append({
        "question": "What are the column names in this dataset?",
        "expects":  [col.lower() for col in df.columns[:3]],
        "label":    "column names",
    })

    cases.append({
        "question": "Are there any duplicate rows?",
        "expects":  ([str(dup_count), "yes", "duplicate"] if dup_count > 0
                     else ["no", "0", "none"]),
        "label":    "duplicate rows",
    })

    cases.append({
        "question": "How many total missing values are in the dataset?",
        "expects":  [str(total_nulls)],
        "label":    "total null count",
    })

    cases.append({
        "question": "Is there a column called fake_column_xyz in this dataset?",
        "expects":  ["no", "not", "doesn't", "does not"],
        "label":    "fake column honesty check",
    })

    # --- Null column questions ---

    null_col, null_count = get_null_info(df)
    if null_col:
        cases.append({
            "question": "Which column has the most missing values?",
            "expects":  [null_col.lower(), null_col],
            "label":    "most null column",
        })
        cases.append({
            "question": f"How many null values are in the {null_col} column?",
            "expects":  [str(null_count)],
            "label":    "null count in column",
        })

    # --- Numeric column questions (up to 3 columns) ---

    for col in num_cols[:3]:
        col_label = col.replace("_", " ")
        avg       = round(df[col].mean(), 1)
        max_val   = int(df[col].max())
        min_val   = int(df[col].min())

        cases.append({
            "question": f"What is the average {col_label}?",
            "expects":  [str(int(avg)), f"{int(avg):,}"],
            "label":    f"average {col_label}",
        })
        cases.append({
            "question": f"What is the highest {col_label} in the dataset?",
            "expects":  [str(max_val), f"{max_val:,}"],
            "label":    f"max {col_label}",
        })
        cases.append({
            "question": f"What is the lowest {col_label} in the dataset?",
            "expects":  [str(min_val), f"{min_val:,}"],
            "label":    f"min {col_label}",
        })

    # --- Category column questions (up to 2 columns) ---

    for col in text_cols[:2]:
        col_label = col.replace("_", " ")
        top_value = str(df[col].value_counts().idxmax())
        unique_count = int(df[col].nunique())

        cases.append({
            "question": f"What is the most common value in the {col_label} column?",
            "expects":  [top_value.lower(), top_value],
            "label":    f"top value in {col_label}",
        })
        cases.append({
            "question": f"How many unique values are in the {col_label} column?",
            "expects":  [str(unique_count)],
            "label":    f"unique count in {col_label}",
        })

    return cases


def run_case(agent, case, index):
    """Run one eval case and return the result."""
    start = time.monotonic()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": case["question"]}]},
            config={"configurable": {"thread_id": f"eval-{index}"}},
        )
        answer = extract_answer(result)
        failed = False
    except Exception as error:
        answer = f"(failed: {error})"
        failed = True

    elapsed = round(time.monotonic() - start, 1)
    passed  = not failed and any(
        e.lower() in answer.lower() for e in case["expects"]
    )

    return {"question": case["question"], "label": case["label"],
            "answer": answer, "passed": passed, "elapsed": elapsed}


def main():
    """Load a file, generate eval cases from it, run them, and print results."""
    print("=" * 40)

    df, path = load_file_with_retry("\nEnter path to CSV or Excel file (or 'exit' to quit): ")
    set_active_dataframe(df)

    cases = build_eval_cases(df)
    print(f"\nLoaded '{path}' — {len(df)} rows, {len(df.columns)} columns")
    print(f"Generated {len(cases)} eval questions from this file.\n")

    try:
        agent = build_agent()
    except RuntimeError as error:
        print(f"Could not start agent: {error}")
        sys.exit(1)

    results = [run_case(agent, case, i) for i, case in enumerate(cases)]

    print()
    for i, r in enumerate(results, 1):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{i}/{len(results)}] {status} ({r['elapsed']}s) — {r['label']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer']}\n")

    passed = sum(r["passed"] for r in results)
    total  = len(results)
    secs   = sum(r["elapsed"] for r in results)

    print("=" * 40)
    print(f"Result: {passed}/{total} passed in {secs:.1f}s")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
