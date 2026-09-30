import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
import os

from src import Evaluation, KBManager


load_dotenv()
token = os.getenv("HF_TOKEN")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "Qwen/Qwen3-8B"
model_name = "Qwen/Qwen3.8-27B"
evaluation = Evaluation(
    model_name=model_name,
    device="cuda",
    max_new_tokens=8192, 
    hf_token=token,
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
    results.append(evaluation.evaluate_kb(kbd=kb_data))
    kbds.append(kb_data)

df_results, df_predictions = evaluation.results_to_df(
    kbds,
    results,
    preds_outpath=f"./exports/qwen25_preds{0}.xlsx",
    results_outpath=f"./exports/qwen25_results{0}.xlsx",
)
df_steps = evaluation.score_by_reasoning_steps(df_predictions)


print("END")
