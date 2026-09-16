import numpy as np 
from dataclasses import dataclass 
from typing import List, Optional, Literal, Any 
from numpy.typing import NDArray


@dataclass
class KBData: 
    qid: int 
    query:str 
    relevant_observation: NDArray 
    prompt: str 
    all_kb: str 


@dataclass
class Results: 
    accuracy: float 
    f1_macro: float 
    f1: NDArray 
    preds: NDArray
    trues: NDArray