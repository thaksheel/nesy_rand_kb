import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import os
import warnings 

from src import (
    Evaluation,
    KBManager,
    OpenModelSelection,
    CloseModelSelection,
    Methods,
    Prompt,
    AnswerFormat,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*_check_is_size.*",
)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
oa_token = os.getenv("OPENAI_TOKEN")
model_name = CloseModelSelection.gpt6_luna
model_name = OpenModelSelection.qwen3_8b
DIR = "./exports/w44/"

# for provider in ["openai", "hf"]:
for provider in ["hf"]:
    print(f"----> Running {provider}")
    if provider == "openai": 
        model_name = CloseModelSelection.gpt6_luna
        temp = 1 
    else:
        model_name = OpenModelSelection.qwen3_8b
        temp = 0.6 
    evaluation = Evaluation(
        provider=provider,
        model_name=model_name.value,
        token=oa_token,
        max_new_tokens=16384,
        temperature=temp,
        load_4bit=True,
        log=False, 
    )
    for method in Methods:
        p = Prompt(
            method=method,
            num_examples=10,
            answer_format=AnswerFormat.REASONING,
        )
        results, kbds = [], []
        for i in range(5):
            kbm = KBManager(
                kb_path=f"./data/new/facts/rand_kb{i+1}.txt",
                instruction_path="./data/ins.txt",
                reference_path="./data/reference.txt",
                queries_data_path=f"./data/new/queries/all_boolean_queries{i+1}.txt",
            )
            kb_data = kbm.get_kb_data(add_reference=True, prompt=p)
            kbds.append(kb_data)
            results.append(
                evaluation.evaluate_kb(
                    kbd=kb_data,
                    reasoning_outpath=f"{DIR + model_name.name}_{p.name}_{provider}_reasoning{i}.csv",
                )
            )
        print(f"-----> Done: {p.name}_{provider}")
        df_results, df_predictions = evaluation.results_to_df(
            kbds,
            results,
            preds_outpath=f"{DIR}{model_name.name}_{p.name}_{provider}_preds.xlsx",
            results_outpath=f"{DIR}{model_name.name}_{p.name}_{provider}_results.xlsx",
        )
        df_steps = evaluation.score_by_reasoning_steps(df_predictions)
        df_steps.to_excel(f"{DIR}{model_name.name}_{p.name}_{provider}_scores_steps.xlsx")

print("END")
