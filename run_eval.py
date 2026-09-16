import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.manager import KBManager
from src.evaluation import Evaluation

for ref in [True, False]:
    results = []
    for i in range(5):
        kbm = KBManager(
            facts_path=f"./data/all_facts{i+1}.txt",
            kb_path=f"./data/rand_kb{i+1}.txt",
            instruction_path="./data/ins.txt",
            queries_path=f"./data/test_queries{i+1}.txt",
            reference_path="./data/reference.txt",
        )
        kb_data = kbm.get_kb_data(add_reference=ref)
        modelname = "meta-llama/Llama-3.1-8B-Instruct"
        modelname = "Qwen/Qwen2.5-7B-Instruct"
        evaluation = Evaluation(
            model_name=modelname, 
            device="cuda", 
        )
        custom_truth = np.ones(len(kb_data))
        result = evaluation.evaluate_kb(kbd=kb_data, custom_truth=custom_truth)
        df_raw = pd.DataFrame({"preds": result.preds, "trues": result.trues})
        r = "w" if ref else "wo"
        df_raw.to_excel(f"./exports/qwne25_kb{i}_{r}_ref_all.xlsx")
        print(
            f"\n\naccuracy={result.accuracy:.4f} "
            f"f1_macro={result.f1_macro:.4f} "
            f"f1={result.f1} "
        )
        results.append(result)
    df_results = pd.DataFrame([d.__dict__ for d in results])
    df_results.to_excel(f"./exports/qwen25_rslt_{r}_ref.xlsx")

print("END")