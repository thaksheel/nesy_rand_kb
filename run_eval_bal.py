import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
import os

from src import Evaluation, KBManager

load_dotenv()
token = os.getenv("HF_TOKEN")
evaluation = Evaluation(
    model_name=model_name,
    device="cuda",
    max_new_tokens=4094,
    hf_token=token,
    load_4bit=False,
)
results = []
kbds = []
for i in range(1):
    kbm = KBManager(
        kb_path=f"./data/kbs/rand_kb{i+1}.txt",
        instruction_path="./data/ins.txt",
        reference_path="./data/reference.txt",
        queries_data_path=f"./data/queries/queries{i+1}.txt",
    )
    kb_data = kbm.get_kb_data(add_reference=True)
    results.append(
        evaluation.evaluate_kb(
            kbd=kb_data,
            reasoning_outpath=f"./exports/llama31_reasoning{i}.csv",
        )
    )
    kbds.append(kb_data)

df_results, df_predictions = evaluation.results_to_df(
    kbds,
    results,
    preds_outpath=f"./exports/llama31_preds{0}.xlsx",
    results_outpath=f"./exports/llama31_results{0}.xlsx",
)
df_steps = evaluation.score_by_reasoning_steps(df_predictions)
df_steps.to_excel(f"./exports/scores_steps{0}.xlsx")

print("END")
