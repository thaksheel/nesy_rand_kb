import re
import numpy as np 
import clingo

from src.manager import KBManager


def kb_to_asp(kb_lines):
    asp = []
    for line in kb_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.endswith("."):
            line += "."
        asp.append(line)
    return "\n".join(asp)


def split_by_top_level_comma(s: str):
    """Split a string by commas, but only at the top level (not inside parentheses)."""
    parts = []
    current = []
    paren_depth = 0

    for char in s:
        if char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth -= 1
            current.append(char)
        elif char == "," and paren_depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def query_to_constraints(query_str):
    atoms = split_by_top_level_comma(query_str.strip().rstrip("."))
    # Normalize predicate names
    normalized = []
    for atom in atoms:
        atom = atom.strip()
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", atom)
        pred = m.group(1).lower()
        args = m.group(2)
        normalized.append(f"{pred}({args})")
    # Build safe rule
    body = ", ".join(normalized)
    rule = f"query_satisfied :- {body}."
    # Constraint requiring the query to be true
    constraint = ":- not query_satisfied."
    return rule + "\n" + constraint


def evaluate_query(kb_lines, query_str) -> bool:
    """Return True iff KB ∧ query is satisfiable under Clingo."""
    asp_program = kb_to_asp(kb_lines)
    asp_constraints = query_to_constraints(query_str)

    ctl = clingo.Control(["--warn=no-atom-undefined"])
    ctl.add("base", [], asp_program)
    ctl.add("query", [], asp_constraints)

    try:
        ctl.ground([("base", []), ("query", [])])
    except RuntimeError as e:
        print("Clingo parsing failed:")
        print("  KB/constraints caused error:", e)
        return False

    result = ctl.solve()
    return result.satisfiable

shapes = []
for i in range(5):
    kbm = KBManager(
        facts_path=f"./data/all_facts{i+1}.txt",
        kb_path=f"./data/rand_kb{i+1}.txt",
        instruction_path="./data/ins.txt",
        queries_path=f"./data/test_queries{i+1}.txt",
        reference_path="./data/reference.txt",
    )
    kb_data = kbm.get_kb_data(add_reference=True)
    results = []
    for kb in kb_data:
        is_true = evaluate_query(kb.relevant_observation, kb.query)
        results.append(is_true)
    results = np.array(results)
    print(f"--> proportion={results.sum()}/{results.shape[0]}")
print("END")
