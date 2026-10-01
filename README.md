# Neuro-Symbolic LLM Benchmark

This repository evaluates LLMs on neuro-symbolic reasoning datasets. The goal is to build benchmark datasets that are difficult to answer through memorization and use model evaluations to measure that challenge. The pipeline currently supports Hugging Face and OpenAI models through LangChain; additional providers can be added in `src/llm_provider.py`.

## Repository Structure

- `data/`: Knowledge bases, query sets, instructions, reference examples, and related dataset files.
- `src/`: Prompt/data management, LLM provider integration, and evaluation and scoring logic.
- `run_eval_bal.py`: Current evaluation entry point; runs evaluation across five datasets and exports reasoning, predictions, results, and depth-based scores.
- `run_eval.py`, `run_kb.py`, `run_test.py`, `test_kb.py`: Additional and earlier pipeline/test scripts.

## Run an Evaluation

Set the applicable `OPENAI_TOKEN` or `HF_TOKEN` in a local `.env` file, choose the matching model and provider in `run_eval_bal.py`, then run:

```bash
python run_eval_bal.py
```

The script currently selects GPT-6 Luna (`provider="openai"`); the Qwen selection immediately above it is overridden. The script writes per-dataset reasoning CSVs and combined prediction, result, and reasoning-depth score spreadsheets under `exports/`.

## Preliminary Results

| Dataset | Qwen2.5 7B Accuracy | Qwen2.5 7B F1 Macro | GPT-6 Luna Accuracy | GPT-6 Luna F1 Macro |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 57.5% | 48.1% | 100.0% | 100.0% |
| 1 | 52.5% | 42.0% | 100.0% | 100.0% |
| 2 | 50.0% | 37.3% | 85.0% | 84.7% |
| 3 | 62.5% | 58.1% | 97.5% | 97.5% |
| 4 | 57.5% | 50.5% | 80.0% | 79.2% |