import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src import Evaluation, KBManager

results = []
for i in range(5):
    kbm = KBManager(
        facts_path=f"./data/facts/all_facts{i+1}.txt",
        kb_path=f"./data/kbs/rand_kb{i+1}.txt",
        instruction_path="./data/ins.txt",
        reference_path="./data/reference.txt",
        queries_data_path=f"./data/queries/queries{i+1}.txt",
    )
    kb_data = kbm.get_kb_data(add_reference=True)
    modelname = "Qwen/Qwen2.5-7B-Instruct"
    modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    evaluation = Evaluation(
        model_name=modelname,
        device="cuda",
    )
    rslt = evaluation.evaluate_kb(kbd=kb_data, custom_truth=None, use_all_kb=True)
    df_raw = pd.DataFrame({"preds": rslt.preds, "trues": rslt.trues})
df_results = pd.DataFrame([d.__dict__ for d in results])
df_results.to_excel(f"./exports/tllama_rslt0.xlsx")


print("END")
