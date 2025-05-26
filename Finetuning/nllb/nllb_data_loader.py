import json
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def load_tokenized_dataset(model_name="/home/mlt_ml2/ML_Applied_Project_2025S/Finetuning/nllb/nllb_results_run3/checkpoint-500", train_size=800, val_size=100, test_size=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "deu_Latn"

    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in tqdm(f, desc=f"Loading {path}")]
        return lines

    def tokenize_pairs(pairs):
        src = [pair['translation']['en'] for pair in pairs]
        tgt = [pair['translation']['de'] for pair in pairs]
        tokenized = tokenizer(src, text_target=tgt, padding=True, truncation=True)
        return Dataset.from_dict(tokenized)

    train_pairs = load_jsonl("/home/mlt_ml2/ML_Applied_Project_2025S/Data/train.clean.jsonl")
    val_pairs = load_jsonl("/home/mlt_ml2/ML_Applied_Project_2025S/Data/val.clean.jsonl")

    train_dataset = tokenize_pairs(train_pairs)
    val_dataset = tokenize_pairs(val_pairs)

    return train_dataset, val_dataset
