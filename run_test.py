from dotenv import load_dotenv
import os 

from src import LLMProvider

load_dotenv()
token = os.getenv("OPENAI_TOKEN")
model_name = "gpt-6-luna"
lp = LLMProvider(
    provider="openai",
    token=token,
    model_name=model_name,
    temperature=1,
)
messages = [{"role": "user", "content": "why is the sky not orange?"}]
res = lp.invoke(messages=messages, prompt=None)
print(res)

token = os.getenv("HF_TOKEN")
model_name = "Qwen/Qwen2.5-7B-Instruct"
lp = LLMProvider(
    provider="hf",
    token=token,
    model_name=model_name,
    temperature=0,
)
messages = [{"role": "user", "content": "why is the sky not orange?"}]
res = lp.invoke(messages=messages, prompt=None)
print(res)
print("END \n")
