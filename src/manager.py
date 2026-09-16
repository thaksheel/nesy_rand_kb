import numpy as np
import re
import os
from typing import List
from copy import deepcopy
from numpy.typing import NDArray

from .config import KBData


class KBManager:
    def __init__(
        self,
        facts_path: str,
        kb_path: str,
        instruction_path: str,
        queries_path: str,
        reference_path: str,
    ):
        self.all_facts: List[str] = self.load_kb(facts_path)
        self.kb: List[str] = self.load_kb(kb_path)
        self.knowledge: List[str] = self.kb + self.all_facts
        self.instructions_template = self.load_txt(instruction_path)
        self.queries = self.load_kb(queries_path)
        self.reference = self.load_txt(reference_path)

    def extract_predicates(self, query: str):
        return set(re.findall(r"([a-zA-Z0-9_]+)\(", query))

    def extract_constants(self, query: str):
        return set(re.findall(r"\b(a[0-9]+)\b", query))

    def extract_variables(self, query: str):
        return set(re.findall(r"\b(X[0-9]+)\b", query))

    def load_kb(self, path: str):
        """Load kb from a txt file each query on a new line ending with fullstop."""
        with open(path) as f:
            kb = [line.strip() for line in f if line.strip()]
        return kb

    def load_txt(self, path: str):
        with open(path) as f:
            return f.read()

    def relevant_observation(self, query: str) -> NDArray:
        predicates = self.extract_predicates(query)
        constants = self.extract_constants(query)
        relevant = []
        for line in self.knowledge:
            line_preds = set(re.findall(r"([a-zA-Z0-9_]+)\(", line))
            if not (line_preds & predicates):
                continue
            lines_consts = set(re.findall(r"\b(a[0-9]+)\b", line))
            if constants and not (lines_consts & constants) and lines_consts:
                continue
            relevant.append(line)
        return np.array(relevant)

    def generate_query_prompt(self, query: str):
        kb_observation = self.relevant_observation(query)
        kb_observation = " ".join(kb_observation.tolist())
        prompt = deepcopy(self.instructions_template)
        prompt = prompt.replace("<KB_REV/>", kb_observation)
        prompt = prompt.replace("<QUERY/>", query)
        return prompt

    def generate_query_all(self, query: str):
        kb_observation = self.knowledge
        kb_observation = " ".join(kb_observation)
        prompt = deepcopy(self.instructions_template)
        prompt = prompt.replace("<KB_REV/>", kb_observation)
        prompt = prompt.replace("<QUERY/>", query)
        return prompt

    def add_reasoning_ref(self, kb: List[KBData]):
        for i, k in enumerate(kb):
            kb[i].prompt = kb[i].prompt.replace("<RESONING EXAMPLE/>", self.reference)
        return kb

    def get_kb_data(
        self,
        add_reference: bool = False,
    ) -> List[KBData]:
        kb = [
            KBData(
                qid=i,
                query=q,
                relevant_observation=self.relevant_observation(q),
                prompt=self.generate_query_prompt(q),
                all_kb=self.generate_query_all(q),
            )
            for i, q in enumerate(self.queries)
        ]
        if add_reference:
            kb = self.add_reasoning_ref(kb)
        return kb
