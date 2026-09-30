import numpy as np 
from dataclasses import dataclass 
from typing import List, Optional, Literal, Any 
from numpy.typing import NDArray
from enum import Enum 


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
    input_token_avg: int 
    input_token: NDArray
    output_token: int 
    output_token_avg: NDArray


@dataclass
class LLMOut: 
    response: str 
    input_token:int 
    output_token: int 
    model_name: str 
    provider: str 


class ModelSelection(Enum):
    tllama = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    qwen38_27b = "Qwen/Qwen3.8-27B"
    llama31_8b = "meta-llama/Llama-3.1-8B-Instruct" 
    qwen25_7b = "Qwen/Qwen2.5-7B-Instruct" 
    qwen3_8b = "Qwen/Qwen3-8B"
