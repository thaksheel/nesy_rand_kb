import re
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


class Methods(Enum):
    ZERO_SHOT = 0
    ZERO_SHOT_COT = 1
    FEW_SHOT = 2
    FEW_SHOT_COT = 3

    @property
    def is_cot(self) -> bool:
        return self in (Methods.ZERO_SHOT_COT, Methods.FEW_SHOT_COT)

    @property
    def is_few_shot(self) -> bool:
        return self in (Methods.FEW_SHOT, Methods.FEW_SHOT_COT)


class AnswerFormat(Enum):
    """Output format; both are parsed by Evaluation.extract_results."""

    ANSWER = (
        "Respond with only a JSON object and nothing else: "
        '{"answer": true} or {"answer": false}.'
    )
    REASONING = (
        'Respond with only a JSON object and nothing else, writing "reasoning" before '
        '"answer": {"reasoning": "<your step-by-step derivation>", "answer": true or false}.'
    )


COT = "Let's work this out in a step by step way to ensure we have the right answer."

TASK = """You answer queries over a knowledge base of facts and rules written in Datalog.
{syntax}

A query is true if it can be derived from the facts by applying the rules any number of times, and false otherwise: anything that cannot be derived is false. If the query contains variables, it is true when some assignment of constants to those variables makes it derivable. If it contains several atoms, all of them must hold under the same assignment.

Use only the knowledge base. Do not assume any fact that is neither stated nor derivable."""

_ATOM = re.compile(r"[A-Za-z0-9_]+\(([^()]*)\)")


def _args(text: str) -> List[str]:
    return [
        a.strip()
        for m in _ATOM.finditer(text)
        for a in m.group(1).split(",")
        if a.strip()
    ]


def predicates(text: str) -> set:
    return set(re.findall(r"([A-Za-z0-9_]+)\(", text))


def constants(text: str) -> set:
    return {a for a in _args(text) if not (a[0].isupper() or a[0] == "_")}


def syntax_lines(knowledge: List[str]) -> str:
    """Explain facts and rules with the first fact and first rule of this KB,
    so the explanation always matches the KB's own syntax."""
    fact = next(
        (
            l
            for l in knowledge
            if len(_ATOM.findall(l)) == 1 and set(_args(l)) == constants(l)
        ),
        None,
    )
    rule = next((l for l in knowledge if len(_ATOM.findall(l)) >= 2), None)
    lines = []
    if fact:
        lines.append(
            f"- A fact such as {fact} states that a relation holds between constants."
        )
    if rule:
        lines.append(
            f"- A rule such as {rule} means that its conclusion holds for any values of its "
            "variables that make all of its conditions hold."
        )
    return "\n".join(lines)


def _pick(
    order: List[int], labels: List[bool], amount: int, balanced: bool
) -> List[int]:
    """Take `amount` indices following `order`; if balanced, split true/false evenly
    (the extra one for odd amounts goes to the label of the first index in `order`)."""
    amount = min(amount, len(order))
    if amount <= 0:
        return []
    if not balanced:
        return order[:amount]
    first = labels[order[0]]
    want = {first: (amount + 1) // 2, not first: amount // 2}
    picked = []
    for i in order:
        if want[labels[i]] > 0:
            picked.append(i)
            want[labels[i]] -= 1
    for i in order:  # one label ran out: fill with the next ones in order
        if len(picked) >= amount:
            break
        if i not in picked:
            picked.append(i)
    return picked


class ShotsExamples:
    """Random examples from the other queries of the same KB, never the target query."""

    nearest_last = False

    def __init__(self, queries: List[str], groundtruths: List[bool]):
        self.queries = queries
        self.groundtruths = groundtruths
        self.examples: List[Tuple[int, str, bool]] = []

    def candidates(self, qid: int, exclude_shared_constants: bool) -> List[int]:
        target = self.queries[qid]
        shared = constants(target) if exclude_shared_constants else set()
        return [
            i
            for i, q in enumerate(self.queries)
            if q != target and not (shared & constants(q))
        ]

    def order(self, qid: int, cands: List[int], rng: np.random.Generator) -> List[int]:
        return [int(i) for i in rng.permutation(cands)]

    def get_examples(
        self,
        qid: int,
        amount: int,
        rng: np.random.Generator,
        balanced: bool = True,
        exclude_shared_constants: bool = True,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ) -> List[Tuple[int, str, bool]]:
        cands = self.candidates(qid, exclude_shared_constants)
        order = self.order(qid, cands, rng) if cands else []
        rank = {i: r for r, i in enumerate(order)}
        picked = _pick(order, self.groundtruths, amount, balanced)
        picked.sort(key=lambda i: -rank[i] if self.nearest_last else rank[i])
        examples = []
        for i in picked:
            answer = self.groundtruths[i]
            if demonstration and rng.random() < demon_rate:
                answer = bool(
                    rng.integers(2)
                )  # random label, as in the sentiment version
            examples.append((i, self.queries[i], answer))
        self.examples = examples
        return examples


class KATEShotsExamples(ShotsExamples):
    """KATE-style: the queries most similar to the target, by shared predicates (Jaccard)
    instead of BERT embeddings. Ties are broken at random; the most similar example is
    shown last, next to the target query."""

    nearest_last = True

    def order(self, qid: int, cands: List[int], rng: np.random.Generator) -> List[int]:
        target = predicates(self.queries[qid])
        sim = []
        for i in cands:
            p = predicates(self.queries[i])
            sim.append(len(target & p) / max(len(target | p), 1))
        jitter = rng.random(len(cands))
        return [cands[j] for j in np.lexsort((jitter, -np.array(sim)))]


@dataclass
class Prompt:
    method: Methods = Methods.ZERO_SHOT
    answer_format: Optional[AnswerFormat] = (
        None  # None: REASONING for CoT methods, else ANSWER
    )
    num_examples: int = 4
    few_shots_type: Literal["random", "kate"] = "random"
    balanced: bool = True  # half true, half false examples
    exclude_shared_constants: bool = (
        True  # skip examples that mention the target's constants
    )
    demonstration: bool = False  # randomize example labels with probability demon_rate
    demon_rate: float = 0.5
    seed: int = 0

    @property
    def fmt(self) -> AnswerFormat:
        if self.answer_format is not None:
            return self.answer_format
        return AnswerFormat.REASONING if self.method.is_cot else AnswerFormat.ANSWER

    @property
    def name(self) -> str:
        """Short label for file names, e.g. few_shot_cot-k4_random."""
        name = self.method.name.lower()
        if self.method.is_few_shot:
            name += f"-k{self.num_examples}_{self.few_shots_type}"
            if self.demonstration:
                name += f"_demo{self.demon_rate:g}"
        return name


class PromptStrategy:
    def __init__(
        self,
        prompt: Prompt,
        knowledge: List[str],
        queries: List[str],
        groundtruths: List[bool],
        kb_name: str = "kb",
        reference: str = "",
    ):
        self.prompt = prompt
        self.knowledge = knowledge
        self.queries = queries
        self.kb_name = kb_name
        self.reference = reference
        selector = (
            KATEShotsExamples if prompt.few_shots_type == "kate" else ShotsExamples
        )
        self.shots = selector(queries, groundtruths)

    def _instructions(self) -> str:
        out = TASK.format(syntax=syntax_lines(self.knowledge))
        if self.reference:
            out += f"\n\nExample of the expected reasoning:\n{self.reference.strip()}"
        return out

    def _kb(self) -> str:
        return "Knowledge base:\n" + "\n".join(self.knowledge)

    def _ending(self, query: str) -> str:
        cot = f"{COT}\n" if self.prompt.method.is_cot else ""
        return f"Query: {query}\n\n{cot}{self.prompt.fmt.value}"

    def zero_shot(self, qid: int) -> Dict[str, str]:
        user = f"{self._kb()}\n\n{self._ending(self.queries[qid])}"
        return {"system": self._instructions(), "user": user}

    def few_shots(self, qid: int) -> Dict[str, str]:
        p = self.prompt
        rng = np.random.default_rng(
            [p.seed, zlib.crc32(f"{self.kb_name}/{qid}".encode())]
        )
        examples = self.shots.get_examples(
            qid,
            p.num_examples,
            rng,
            balanced=p.balanced,
            exclude_shared_constants=p.exclude_shared_constants,
            demonstration=p.demonstration,
            demon_rate=p.demon_rate,
        )
        block = ""
        if examples:
            lines = "\n".join(f"{q}: {str(a).lower()}" for _, q, a in examples)
            block = f"\n\nExample queries over this knowledge base, with their correct answers:\n{lines}"
        user = f"{self._kb()}{block}\n\n{self._ending(self.queries[qid])}"
        return {"system": self._instructions(), "user": user}

    def _build(self, qid: int) -> Dict[str, str]:
        return (
            self.few_shots(qid)
            if self.prompt.method.is_few_shot
            else self.zero_shot(qid)
        )

    def get_prompt(self, qid: int) -> str:
        obj = self._build(qid)
        return obj["system"] + "\n\n" + obj["user"]

    def get_system_prompt(self, qid: int) -> str:
        return self._build(qid)["system"]

    def get_user_prompt(self, qid: int) -> str:
        return self._build(qid)["user"]
