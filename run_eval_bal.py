import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src import Evaluation, KBManager

model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
evaluation = Evaluation(
    model_name=model_name,
    device="cuda",
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
    results.append(evaluation.evaluate_kb(kbd=kb_data))
    kbds.append(kb_data)

df_results, df_predictions = evaluation.results_to_df(
    kbds,
    results,
    preds_outpath=f"./exports/tllama_preds{1}.xlsx",
    results_outpath=f"./exports/tllama_results{1}.xlsx",
)

print("END")
