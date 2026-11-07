from src import KBManager, Prompt, Methods, AnswerFormat

kbm = KBManager(
    kb_path="./data/kbs/rand_kb1.txt",
    queries_data_path="./data/queries/queries1.txt",
    reference_path="./data/reference.txt",
    instruction_path="./data/ins.txt",
)
prompts = [
    Prompt(
        method=m,
        answer_format=AnswerFormat.REASONING,
        num_examples=10,
        few_shots_type="random",
    )
    for m in Methods
]
paths = kbm.save_prompts(prompts, outdir="./exports/prompts", qids=[0])
print("\n".join(paths))
