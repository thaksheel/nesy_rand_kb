import numpy as np 
from dataclasses import dataclass 
from typing import List, Optional, Literal, Any 
from numpy.typing import NDArray


@dataclass
class KBData: 
    qid: int 
    query:str 
    prompt: str 
    groundtruth: bool
    depths: int 


@dataclass
class Results: 
    accuracy: float 
    f1_macro: float 
    f1: NDArray 
    preds: NDArray
    trues: NDArray 

