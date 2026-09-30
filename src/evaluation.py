import torch
from typing import Tuple, List, Dict, Optional, Literal
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging

from . import Results, KBData, LLMResponse


class Evaluation:
    def __init__(
        self,
        model_name: str,
        device: Literal["cpu", "cuda"],
        hf_token: str, 
        max_new_tokens: int = 2048,
        log: bool = True, 
        logspath: str = "./exports/logs.out"
    ):
        self.log = log
        self.llm = LLMResponse(
            model_name=model_name,
            hf_token=hf_token,
            max_new_tokens=max_new_tokens,
            device=torch.device(device),
        ) 
        logging.basicConfig(
            filename=logspath,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get_groundtruth(self, kbd: List[KBData]):
        return np.array([k.groundtruth for k in kbd]).astype(int)

    def extract_results(self, text: str) -> Optional[Tuple[bool, str]]:
        reasoning = re.search(r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        answer = re.search(
            r'"answer"\s*:\s*(true|false|"[^"]*"|\d+(?:\.\d+)?)', text, re.IGNORECASE
        )
        reasoning = reasoning.group(1) if reasoning else None
        answer = answer.group(1) == "True" if answer else None
        return answer, reasoning

    def save_reasoning(self, kbd: List[KBData], reasonings: List[str], outpath: str):
        fields = ["qid", "query", "groundtruth", "depths", "reasoning"]
        data = dict(
            zip(
                [k for k in fields],
                [[] for _ in fields],
            )
        )
        for i, kb in kbd:
            for k in data.keys():
                if k == "reasoning":
                    data[k].append(reasonings[i])
                    continue
                data[k].append(getattr(kb, k))
        df = pd.DataFrame(data)
        df.to_csv(outpath)
        return df

    def evaluate_kb(
        self,
        kbd: List[KBData],
        reasoning_outpath: str = None,
    ) -> Results:
        preds = []
        reasonings = []
        for i, kb in enumerate(tqdm(kbd)):
            gs = self.get_groundtruth(kbd)
            out = self.llm.response([{"role": "user", "content": kb.prompt}])
            pred, reasoning = self.extract_results(out)
            if pred is None:
                raise ValueError(
                    "predictions/reasoning from llm is none in evaluate_kb."
                )
            preds.append(pred)
            reasonings.append(reasoning)
            if self.log:
                s = (
                    f"id={kb.qid} "
                    f"preds={pred} "
                    f"true={gs[i]} "
                    f"reasoning={reasoning} "
                )
                logging.info(s)
        if reasoning_outpath:
            self.save_reasoning(kbd, reasonings)
        preds = np.array(preds).astype(int)
        return Results(
            accuracy=accuracy_score(gs, preds),
            f1_macro=f1_score(gs, preds, average="macro"),
            f1=f1_score(gs, preds, average=None),
            trues=np.array(gs).astype(int),
            preds=np.array(preds).astype(int),
        )

    def score_by_reasoning_steps(
        self, df: pd.DataFrame
    ):
        metrics_by_depth = (
            df.groupby("depths")
            .apply(
                lambda g: pd.Series(
                    {
                        "accuracy": accuracy_score(g["groundtruth_num"], g["preds"]),
                        "f1_macro": f1_score(
                            g["groundtruth_num"], g["preds"], zero_division=0, average="macro"
                        ),
                        "f1": f1_score(
                            g["groundtruth_num"], g["preds"], zero_division=0, average=None
                        ),
                        "count": len(g),
                    }
                )
            )
            .reset_index()
        )
        return metrics_by_depth

    def results_to_df(
        self,
        kbds: List[List[KBData]],
        results: List[Results],
        preds_outpath: str = None,
        results_outpath: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_results = pd.DataFrame([r.__dict__ for r in results])
        df_results: pd.DataFrame = df_results.drop(columns=["preds", "trues"])
        df_predictions = []
        for i, kbd in enumerate(kbds):
            df_p = pd.DataFrame([r.__dict__ for r in kbd])
            df_p["kb_num"] = [i] * len(df_p)
            df_p["preds"] = results[i].preds.astype(int)
            df_p["groundtruth"] = df_p["groundtruth"].astype(int)
            df_predictions.append(df_p)
        df_predictions: pd.DataFrame = pd.concat(df_predictions)
        df_predictions = df_predictions.drop(columns=["prompt"])
        df_results.to_excel(results_outpath)
        df_predictions.to_excel(preds_outpath)
        return df_results, df_predictions
