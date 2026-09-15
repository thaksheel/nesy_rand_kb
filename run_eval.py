import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.manager import KBManager
from src.evaluation import Evaluation

kbm = KBManager(
    facts_path="./data/all_facts1.txt",
    kb_path="./data/rand_kb1.txt",
    instruction_path="./data/ins.txt",
    queries_path="./data/test_queries1.txt",
    reference_path="./data/reference.txt",
)
kb_data = kbm.get_kb_data(add_reference=True)

modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
evaluation = Evaluation(
    model_name=modelname, 
    device="cuda", 
)
results = evaluation.evaluate_kb(kbd=kb_data)
print(results)
print("END")