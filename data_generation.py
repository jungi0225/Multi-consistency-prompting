import pandas as pd

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/tau/commonsense_qa/" + splits["train"])

output_path = "data/CSQA/train.jsonl"
df.to_json(output_path, orient="records", lines=True, force_ascii=False)
print(f"DataFrame has been successfully saved to {output_path}")