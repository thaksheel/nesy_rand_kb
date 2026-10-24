import os
import numpy as np
import re
from typing import List, Optional
from copy import deepcopy
from numpy.typing import NDArray

from . import KBData
from .prompts import Prompt, PromptStrategy


class KBManager:
    def __init__(
        self,
        kb_path: str,
        instruction_path: str,
        queries_data_path: str,
        reference_path: str,
    ):
        """`queries_data_path` needs format [queries, groundtruth, depths]"""
        self.kb_name = os.path.basename(kb_path)
        self.knowledge: List[str] = self.load_kb(kb_path)
        self.instructions_template = self.load_txt(instruction_path)
        self.queries_path = queries_data_path
        self.reference = self.load_txt(reference_path)

        # post init
        self.queries: List[str] = []
        self.groundtruths: List[bool] = []
        self.depths: List[int] = []
        self._load_queries_data()

    def extract_predicates(self, query: str):
        return set(re.findall(r"([a-zA-Z0-9_]+)\(", query))

    def extract_constants(self, query: str):
        return set(re.findall(r"\b(a[0-9]+)\b", query))

    def extract_variables(self, query: str):
        return set(re.findall(r"\b(X[0-9]+)\b", query))

    def load_txt(self, path: str):
        with open(path) as f:
            return f.read()

    def load_kb(self, path: str):
        """Load kb from a txt file each query on a new line ending with fullstop."""
        with open(path) as f:
            kb = [line.strip() for line in f if line.strip()]
        return kb

    def _load_queries_data(self):
        pattern = re.compile(r"^\[(.*?)\]\s*,\s*(YES|NO)\s*,\s*(-?\d+)\.?$")
        with open(self.queries_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = pattern.match(line)
                if not match:
                    print(f"Skipping malformed line: {line}")
                    query, groundtruth, depth = None, None, None
                    continue
                query, groundtruth, depth = match.groups()
                self.queries.append(query)
                self.groundtruths.append(groundtruth == "YES")
                self.depths.append(int(depth))
        return self

    def generate_query_prompt(self, query: str):
        kb_observation = " ".join(self.knowledge)
        prompt = deepcopy(self.instructions_template)
        prompt = prompt.replace("<KB_REV/>", kb_observation)
        prompt = prompt.replace("<QUERY/>", query)
        return prompt

    def add_reasoning_ref(self, kb: List[KBData]):
        for i, _ in enumerate(kb):
            kb[i].prompt = kb[i].prompt.replace("<RESONING EXAMPLE/>", self.reference)
        return kb

    def get_kb_data(
        self,
        add_reference: bool = False,
        prompt: Optional[Prompt] = None,
    ) -> List[KBData]:
        """Without `prompt`: the ins.txt template, as before.
        With `prompt`: zero-shot / few-shot / CoT prompts from prompts.py
        (add_reference then appends reference.txt as an example of the reasoning)."""
        if prompt is not None:
            strategy = PromptStrategy(
                prompt,
                knowledge=self.knowledge,
                queries=self.queries,
                groundtruths=self.groundtruths,
                kb_name=self.kb_name,
                reference=self.reference if add_reference else "",
            )
            return [
                KBData(
                    qid=i,
                    query=q,
                    prompt=strategy.get_prompt(i),
                    groundtruth=self.groundtruths[i],
                    depths=self.depths[i],
                )
                for i, q in enumerate(self.queries)
            ]

        kb = [
            KBData(
                qid=i,
                query=q,
                prompt=self.generate_query_prompt(q),
                groundtruth=self.groundtruths[i],
                depths=self.depths[i],
            )
            for i, q in enumerate(self.queries)
        ]
        if add_reference:
            kb = self.add_reasoning_ref(kb)
        return kb
