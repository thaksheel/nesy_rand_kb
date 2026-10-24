import torch
import warnings
from typing import Tuple, List, Dict, Optional, Literal
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging

from . import Results, KBData, LLMOut, LLMProvider


class Evaluation:

    def __init__(
        self,
        provider: Literal["hf", "openai"],
        model_name: str,
        token: str,
        load_4bit: bool,
        max_new_tokens: int = 2048,
        temperature: float = 1,
        log: bool = True,
        logspath: str = "./exports/logs.out",
    ):
        self.log = log
        self.llm = LLMProvider(
            provider=provider,
            model_name=model_name,
            token=token,
            temperature=temperature,
            load_in_4bit=load_4bit,
            max_new_tokens=max_new_tokens,
        )

        logging.basicConfig(
            filename=logspath,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get_groundtruth(self, kbd: List[KBData]):
        return np.array([k.groundtruth for k in kbd]).astype(int)

    def extract_results(self, text: str) -> Tuple[Optional[bool], Optional[str]]:
        if "<think>" in text and "</think>" not in text:
            return None, None  # thinking never finished
        text = text.split("</think>")[-1]  # parse only the final answer
        reasoning = re.findall(
            r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL
        )
        answer = re.findall(
            r'"answer"\s*:\s*(true|false|"[^"]*"|\d+(?:\.\d+)?)', text, re.IGNORECASE
        )
        reasoning = reasoning[-1] if reasoning else None
        answer = answer[-1].strip('"').lower() in ("true", "yes") if answer else None
        return answer, reasoning

    def save_reasoning(self, kbd: List[KBData], reasonings: List[str], outpath: str):
        fields = ["qid", "query", "groundtruth", "depths", "reasoning"]
        data = dict(
            zip(
                [k for k in fields],
                [[] for _ in fields],
            )
        )
        for i, kb in enumerate(kbd):
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
        trues = []
        reasonings = []
        llm_outs: List[LLMOut] = []
        for i, kb in enumerate(tqdm(kbd)):
            groundtruth = self.get_groundtruth(kbd)
            llm_out = self.llm.invoke([{"role": "user", "content": kb.prompt}])
            pred, reasoning = self.extract_results(llm_out.response)
            if pred is None and reasoning is None:
                warnings.warn("predictions/reasoning from llm is none in evaluate_kb.")
                continue
            preds.append(pred)
            reasonings.append(reasoning)
            llm_outs.append(llm_out)
            trues.append(groundtruth[i])
            if self.log:
                s = (
                    f"id={kb.qid} "
                    f"preds={pred} "
                    f"true={groundtruth[i]} "
                    f"reasoning={reasoning} "
                )
                logging.info(s)
        if reasoning_outpath:
            self.save_reasoning(kbd, reasonings, outpath=reasoning_outpath)
        preds = np.array(preds).astype(int)
        return Results(
            accuracy=accuracy_score(trues, preds),
            f1_macro=f1_score(trues, preds, average="macro"),
            f1=f1_score(trues, preds, average=None),
            trues=np.array(trues).astype(int),
            preds=np.array(preds).astype(int),
            input_token=np.array([l.input_token for l in llm_outs]),
            output_token=np.array([l.output_token for l in llm_outs]),
            input_token_avg=np.array([l.input_token for l in llm_outs]).mean(),
            output_token_avg=np.array([l.output_token for l in llm_outs]).mean(),
        )

    def score_by_reasoning_steps(self, df: pd.DataFrame):
        metrics_by_depth = (
            df.groupby("depths")
            .apply(
                lambda g: pd.Series(
                    {
                        "accuracy": accuracy_score(g["groundtruth"], g["preds"]),
                        "f1_macro": f1_score(
                            g["groundtruth"],
                            g["preds"],
                            zero_division=0,
                            average="macro",
                        ),
                        "f1": f1_score(
                            g["groundtruth"], g["preds"], zero_division=0, average=None
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
        df_results: pd.DataFrame = df_results.drop(
            columns=["preds", "trues", "output_token", "input_token"]
        )
        df_predictions = []
        for i, kbd in enumerate(kbds):
            df_p = pd.DataFrame([r.__dict__ for r in kbd])
            df_p["kb_num"] = [i] * len(df_p)
            df_p["preds"] = results[i].preds.astype(int)
            df_p["input_token"] = results[i].input_token
            df_p["output_token"] = results[i].output_token
            df_p["groundtruth"] = df_p["groundtruth"].astype(int)
            df_predictions.append(df_p)
        df_predictions: pd.DataFrame = pd.concat(df_predictions)
        df_predictions = df_predictions.drop(columns=["prompt"])
        df_results.to_excel(results_outpath)
        df_predictions.to_excel(preds_outpath)
        return df_results, df_predictions
