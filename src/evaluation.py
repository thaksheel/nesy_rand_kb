import torch
import pandas as pd
from typing import Tuple, List, Dict, Optional, Literal
from numpy.typing import NDArray
import numpy as np
import json
import ast 
import re
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
import clingo
from tqdm import tqdm

from .llm_response import LLMResponse
from .manager import KBData
from .config import Results


class Evaluation:
    def __init__(
        self,
        model_name: str,
        device: Literal["cpu", "cuda"],
        max_new_tokens: int = 2048,
    ):
        self.llm = LLMResponse(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=torch.device(device),
        )

    def kb_to_asp(self, kb_lines):
        asp = []
        for line in kb_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not line.endswith("."):
                line += "."
            asp.append(line)
        return "\n".join(asp)

    def split_by_top_level_comma(self, s: str):
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

    def query_to_constraints(self, query_str):
        atoms = self.split_by_top_level_comma(query_str.strip().rstrip("."))
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

    def get_groundtruth(self, relevant_obs, query_str) -> bool:
        """Return True iff KB ∧ query is satisfiable under Clingo."""
        asp_program = self.kb_to_asp(relevant_obs)
        asp_constraints = self.query_to_constraints(query_str)

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

    def extract_json_dict(self, text: str):
        # TODO: improve str_json parsing. Using re to get results for now
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not m:
            m = re.search(r'"answer"\s*:\s*(true|false)', text)
            if not m:
                return None 
            b = m.group(1) == "true"
            return {"answer": b}
        raw = m.group(1)
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        try:
            raw = ast.literal_eval(f"'{raw}'")
        except Exception:
            pass  
        return json.loads(raw.replace("\n", ""))

    def evaluate_kb(self, kbd: List[KBData]):
        gs = []
        preds = []
        for kb in tqdm(kbd):
            gs.append(self.get_groundtruth(kb.relevant_observation, kb.query))
            out = self.llm.response([{"role": "user", "content": kb.prompt}])
            pred = self.extract_json_dict(out)
            if pred is None:
                raise ValueError("predictions from llm is none in evaluate_kb.")
            preds.append(pred["answer"])
        return Results(
            accuracy=accuracy_score(gs, preds),
            f1_macro=f1_score(gs, preds, average="macro"),
            f1=f1_score(gs, preds, average=None),
            trues=np.array(gs).astype(int),
            preds=np.array(preds).astype(int),
        )
