import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os

from src import Evaluation, KBManager, OpenModelSelection, CloseModelSelection


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
oa_token = os.getenv("OPENAI_TOKEN")
model_name = OpenModelSelection.qwen25_7b
model_name = CloseModelSelection.gpt6_luna

evaluation = Evaluation(
    provider="openai",
    model_name=model_name.value,
    token=oa_token,
    max_new_tokens=4092,
    load_4bit=False,
)
results = []
kbds = []
for i in range(5):
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
            reasoning_outpath=f"./exports/{model_name.name}_reasoning{i}.csv",
        )
    )
    kbds.append(kb_data)
df_results, df_predictions = evaluation.results_to_df(
    kbds,
    results,
    preds_outpath=f"./exports/{model_name.name}_preds{0}.xlsx",
    results_outpath=f"./exports/{model_name.name}_results{0}.xlsx",
)
df_steps = evaluation.score_by_reasoning_steps(df_predictions)
df_steps.to_excel(f"./exports/{model_name.name}_scores_steps{0}.xlsx")

print("END")
