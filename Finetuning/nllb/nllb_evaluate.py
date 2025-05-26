import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import evaluate

# Load tokenizer and model from checkpoint
checkpoint_path = "./nllb_results/checkpoint-2502" 
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "deu_Latn"

# Load validation set
raw_dataset = load_from_disk("huggingface_dataset_split")
raw_val = raw_dataset["validation"].select(range(500))

# Evaluation metrics 
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

# Generate translations
def postprocess(texts):
    return [text.strip() for text in texts]

source_texts = raw_val["en"]
references = [[t] for t in raw_val["de"]]

inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
with torch.no_grad():
    generated_tokens = model.generate(**inputs, max_length=200)

preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
preds = postprocess(preds)

# === Compute metrics ===
print("Evaluation:")
print("BLEU:", bleu.compute(predictions=preds, references=references)["score"])
print("chrF:", chrf.compute(predictions=preds, references=references)["score"])
