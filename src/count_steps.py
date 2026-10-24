import argparse
import json
import os
import re
import statistics


# ---------------------------------------------------------------------------
# Aristotle: read the two result files the repo produces
# ---------------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        data = json.load(f)
    return {r["id"]: r for r in data if isinstance(r, dict) and "id" in r}


def steps_in_record(rec):
    """Steps for one path of one problem. Prefers the 'search_round' counter;
    falls back to counting the numbered lines of 'reasoning_step'."""
    if rec is None or "search_round" not in rec and "reasoning_step" not in rec:
        return None  # errored example (repo saves only {'id': ...})
    if "search_round" in rec:
        return int(rec["search_round"])
    return len(re.findall(r"^\d+\. ", rec.get("reasoning_step", ""), flags=re.M))


def _choice(rec):
    c = rec.get("final_choice")
    return c if c in ("A", "B", "D") else "C"


def is_correct(rec_neg, rec_pos):
    """Same rule as the repo's evaluate.py (A=True, B=False, C=Unknown, D=Self-contradictory)."""
    gt = rec_neg.get("ground_truth") or rec_pos.get("ground_truth")
    a1, a2 = _choice(rec_neg), _choice(rec_pos)
    if gt == "A":
        return {a1, a2} in ({"A", "C"}, {"A"})
    if gt == "B":
        return {a1, a2} in ({"B", "C"}, {"B"})
    if gt == "C":
        return a1 == "C" and a2 == "C"
    if gt == "D":
        return {a1, a2} == {"A", "B"}
    return False


def count_aristotle(dataset, model, results_dir="./results"):
    base = os.path.join(results_dir, dataset)
    neg = _load(os.path.join(base, f"{dataset}_{model}_search_negation_True.json"))
    pos = _load(os.path.join(base, f"{dataset}_{model}_search_negation_False.json"))

    rows = []
    for id_ in sorted(set(neg) & set(pos), key=str):
        s_neg, s_pos = steps_in_record(neg[id_]), steps_in_record(pos[id_])
        if s_neg is None or s_pos is None:
            continue
        rows.append(
            {
                "id": id_,
                "neg": s_neg,
                "pos": s_pos,
                "total": s_neg + s_pos,
                "correct": is_correct(neg[id_], pos[id_]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Baselines (CoT-SC, ToT, CR, DetermLR): the repo has no counting code for them,
# so instrument your own runs with StepCounter and save one "num_steps" per problem.
# ---------------------------------------------------------------------------
class StepCounter:
    """Counts visited nodes/steps for one problem.

    ToT / CR / DetermLR: call .visit() each time a thought/node is generated
        (expanded) — that is a "visited node".
    CoT-SC: call .visit() once per sampled chain, or .visit(n) with the number
        of reasoning lines in each chain, depending on which definition you report.
    Or wrap your LLM call with .wrap(fn) to count every call automatically.
    """

    def __init__(self):
        self.n = 0

    def visit(self, k=1):
        self.n += k

    def wrap(self, llm_fn):
        def inner(*a, **kw):
            self.n += 1
            return llm_fn(*a, **kw)

        return inner


def count_baseline(path):
    rows = []
    with open(path) as f:
        text = f.read().strip()
    recs = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(l) for l in text.splitlines() if l.strip()]
    )
    for r in recs:
        rows.append(
            {
                "id": r["id"],
                "total": int(r["num_steps"]),
                "correct": bool(r.get("correct", False)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
def report(rows, label):
    if not rows:
        print("No valid examples found.")
        return
    tot = [r["total"] for r in rows]
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"== {label} ==")
    print(f"Problems counted        : {len(rows)}")
    print(f"Accuracy                : {acc:.2%}")
    print(
        f"Avg steps per problem   : {statistics.mean(tot):.2f}  (std {statistics.pstdev(tot):.2f})"
    )
    if "neg" in rows[0]:
        print(
            f"  avg steps, ¬S_n path  : {statistics.mean(r['neg'] for r in rows):.2f}"
        )
        print(
            f"  avg steps, S_n path   : {statistics.mean(r['pos'] for r in rows):.2f}"
        )
    corr = [r["total"] for r in rows if r["correct"]]
    if corr:
        print(f"Avg steps (correct only): {statistics.mean(corr):.2f}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("aristotle")
    a.add_argument("--dataset", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--results_dir", default="./results")
    a.add_argument("--csv", help="optional per-problem output")
    b = sub.add_parser("baseline")
    b.add_argument("--file", required=True)
    b.add_argument("--csv")
    args = p.parse_args()

    if args.mode == "aristotle":
        rows = count_aristotle(args.dataset, args.model, args.results_dir)
        report(rows, f"Aristotle / {args.dataset} / {args.model}")
    else:
        rows = count_baseline(args.file)
        report(rows, f"Baseline / {os.path.basename(args.file)}")

    if args.csv and rows:
        import csv

        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Per-problem counts written to {args.csv}")


if __name__ == "__main__":
    main()
