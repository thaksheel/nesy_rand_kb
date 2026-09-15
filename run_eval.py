import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.manager import KBManager
from src.evaluation import Evaluation

for i in range(5):
    kbm = KBManager(
        facts_path=f"./data/all_facts{i+1}.txt",
        kb_path=f"./data/rand_kb{i+1}.txt",
        instruction_path="./data/ins.txt",
        queries_path=f"./data/test_queries{i+1}.txt",
        reference_path="./data/reference.txt",
    )
    kb_data = kbm.get_kb_data(add_reference=True)
    modelname = "meta-llama/Llama-3.1-8B-Instruct"
    modelname = "Qwen/Qwen2.5-7B-Instruct"
    evaluation = Evaluation(
        model_name=modelname, 
        device="cuda", 
    )
    results = evaluation.evaluate_kb(kbd=kb_data)
    df_raw = pd.DataFrame({"preds": results.preds, "trues": results.trues})
    df_raw.to_excel(f"./exports/qwne25_kb{i}_w_ref.xlsx")
    print(
        f"accuracy={results.accuracy:.4f} "
        f"f1_macro={results.f1_macro:.4f} "
        f"f1={results.f1} "
    )
print("END")