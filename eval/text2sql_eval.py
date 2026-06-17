"""
Text-to-SQL evaluation harness
==============================
Measures M2's text-to-SQL accuracy with a gold-labelled question set, multiple
trials per question, and automatic correctness scoring (result-set match vs a
hand-written gold query — the standard "execution accuracy" used by Spider/BIRD).

Each (model, question) is run TRIALS times and each attempt is classified as:
  correct       – ran and the result set matches the gold query's result
  wrong         – ran but the result set differs from gold
  exec_error    – generated SQL failed at execution
  guard_reject  – generated SQL was blocked by the SELECT-only safety guard
  llm_fail      – the model didn't return usable SQL

Run:  .venv/bin/python eval/text2sql_eval.py
"""

import os
import sys
import json
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from ai.assistant.text_to_sql import ask, get_connection

TRIALS = 3
MODELS = ["qwen2.5:3b", "qwen2.5:7b"]

# (natural-language question, gold SQL). Gold = the simplest correct query.
CASES = [
    ("2025 年每个月的成交量",
     "SELECT yr_mo, count(*) AS n FROM sold WHERE close_year=2025 GROUP BY yr_mo"),
    ("2026-04 各县按中位成交价从高到低取前 5 个县",
     "SELECT CountyOrParish, median(ClosePrice) AS m FROM sold WHERE yr_mo='2026-04' "
     "GROUP BY CountyOrParish ORDER BY m DESC LIMIT 5"),
    ("Orange 县 2025 年的成交总数",
     "SELECT count(*) AS n FROM sold WHERE CountyOrParish='Orange' AND close_year=2025"),
    ("Los Angeles 县的中位成交价",
     "SELECT median(ClosePrice) AS m FROM sold WHERE CountyOrParish='Los Angeles'"),
    ("2026-04 SingleFamilyResidence 的成交数量",
     "SELECT count(*) AS n FROM sold WHERE yr_mo='2026-04' AND PropertySubType='SingleFamilyResidence'"),
    ("成交量最多的前 3 个县",
     "SELECT CountyOrParish, count(*) AS n FROM sold GROUP BY CountyOrParish ORDER BY n DESC LIMIT 3"),
    ("2025 年各 PropertySubType 的中位成交价",
     "SELECT PropertySubType, median(ClosePrice) AS m FROM sold WHERE close_year=2025 GROUP BY PropertySubType"),
    ("成交量从 2026-03 到 2026-04 增长最多的前 5 个城市",
     "SELECT City FROM (SELECT City, "
     "sum(case when yr_mo='2026-04' then 1 else 0 end) - "
     "sum(case when yr_mo='2026-03' then 1 else 0 end) AS d "
     "FROM sold WHERE yr_mo IN ('2026-03','2026-04') GROUP BY City) ORDER BY d DESC LIMIT 5"),
]


def canon(df):
    """Canonical, order-independent representation of a result set for matching.
    Each row -> a frozenset of its (rounded) cell values, so column naming/order
    don't matter; the whole result -> a frozenset of those rows."""
    if df is None:
        return None
    rows = set()
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            try:
                cells.append(str(round(float(v), 1)))
            except (TypeError, ValueError):
                cells.append(str(v).strip())
        rows.add(frozenset(cells))
    return frozenset(rows)


def classify(res, gold):
    if res["error"]:
        if res["sql"] is None:
            return "llm_fail"
        if "安全校验" in res["error"]:
            return "guard_reject"
        return "exec_error"
    return "correct" if canon(res["df"]) == gold else "wrong"


def main():
    con = get_connection()
    golds = [canon(con.execute(g).df()) for _, g in CASES]
    print(f"Eval: {len(CASES)} questions x {TRIALS} trials x {len(MODELS)} models\n")

    summary = {}
    for model in MODELS:
        tally = Counter()
        per_q = []
        for qi, (q, _) in enumerate(CASES):
            outcomes = []
            for _ in range(TRIALS):
                r = ask(q, con=con, model=model)
                o = classify(r, golds[qi])
                tally[o] += 1
                outcomes.append(o)
            per_q.append(outcomes)
            n_ok = outcomes.count("correct")
            consistent = "=" if len(set(outcomes)) == 1 else "~"
            print(f"  [{model:11}] Q{qi+1}: {n_ok}/{TRIALS} correct {consistent}  "
                  f"({','.join(o[:4] for o in outcomes)})")
        total = len(CASES) * TRIALS
        solved = sum(1 for oc in per_q if oc.count("correct") >= 2)        # majority of trials
        stable = sum(1 for oc in per_q if len(set(oc)) == 1)               # identical across trials
        summary[model] = {
            "accuracy": round(tally["correct"] / total, 3),
            "correct": tally["correct"], "wrong": tally["wrong"],
            "exec_error": tally["exec_error"], "guard_reject": tally["guard_reject"],
            "llm_fail": tally["llm_fail"], "total_trials": total,
            "solved_questions": solved, "n_questions": len(CASES),
            "stable_questions": stable,
        }
        print()

    print("=" * 70)
    for model, s in summary.items():
        print(f"{model}:  accuracy {s['accuracy']*100:.0f}%  "
              f"({s['correct']}/{s['total_trials']} trials)  |  "
              f"solved {s['solved_questions']}/{s['n_questions']} questions  |  "
              f"stable {s['stable_questions']}/{s['n_questions']}")
        print(f"    errors -> wrong:{s['wrong']}  exec:{s['exec_error']}  "
              f"guard:{s['guard_reject']}  llm:{s['llm_fail']}")

    out = os.path.join(BASE_DIR, "outputs", "text2sql_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults -> {out}")
    con.close()


if __name__ == "__main__":
    main()
