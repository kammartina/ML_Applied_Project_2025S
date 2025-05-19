# filepath: /home/mlt_ml2/ML_Applied_Project_2025S/mBART50/mBART50_finetune.py
#import torch
#from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, load_metric
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import set_seed
set_seed(24)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLEU and ROUGE metrics
bleu = load_metric("bleu")
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    """Compute BLEU and ROUGE metrics for Seq2Seq tasks."""
    predictions, labels = eval_pred

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU expects tokenized inputs
    tokenized_preds = [pred.split() for pred in decoded_preds]
    tokenized_labels = [[label.split()] for label in decoded_labels]  # BLEU expects a list of references

    # Compute BLEU
    bleu_score = bleu.compute(predictions=tokenized_preds, references=tokenized_labels)

    # Compute ROUGE
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Return metrics
    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"].mid.fmeasure,
        "rouge2": rouge_score["rouge2"].mid.fmeasure,
        "rougeL": rouge_score["rougeL"].mid.fmeasure,
    }

#accuracy = evaluate.load("accuracy")

#def compute_metrics(eval_pred):
   # """Called at the end of validation. Gives accuracy"""
    #logits, labels = eval_pred
    #predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    #return accuracy.compute(predictions=predictions, references=labels)

# Load the tokenizer and model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    #torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced precision
    attn_implementation="sdpa",  # Use scaled dot-product attention
    device_map="auto"  # Automatically map model to available devices
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


#model_name = "facebook/mbart-large-50-many-to-many-mmt"
#tokenizer = MBart50Tokenizer.from_pretrained(model_name)
#model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set source and target languages
source_lang = "en_XX"  # Source language (English)
target_lang = "de_DE"  # Target language (German)
tokenizer.src_lang = source_lang
model.config.forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]


# Preprocess the dataset
def preprocess_function(examples):
    # Extract the source and target texts from the list of translations
    inputs = [translation["en"] for translation in examples["translation"]]
    targets = [translation["de"] for translation in examples["translation"]]
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=128, truncation=True, padding=True)
    
    # Flatten labels to remove extra nesting
    model_inputs["labels"] = [label for label in labels["input_ids"]]
    
    # Debugging: Print the first few labels to verify the structure
    print("Input IDs:", model_inputs["input_ids"][:5]) 
    print("Labels:", model_inputs["labels"][:5])
    return model_inputs
    

# Load a subset of the training set and the full validation set
dataset = load_dataset(
    "json",
    data_files={
        "train": "/home/mlt_ml2/ML_Applied_Project_2025S/datasplit_json/train.clean.jsonl",
        "validation": "/home/mlt_ml2/ML_Applied_Project_2025S/datasplit_json/val.clean.jsonl"
    },
    split={
        "train": "train[:10%]",  # Load the first 10% of the training set
        "validation": "validation[:10%]"  # oad the first 10% of validation set
    }
)

print(f"Number of training examples: {len(dataset['train'])}")
print(f"Number of validation examples: {len(dataset['validation'])}")

def sample_dataset(dataset, fraction=0.05):
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * fraction)))

dataset["train"] = sample_dataset(dataset["train"], fraction=0.05)
dataset["validation"] = sample_dataset(dataset["validation"], fraction=0.05)

tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=32)
print("Tokenized datasets:", tokenized_datasets["train"][:5])  # Print the first few tokenized examples
# Define a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    logging_steps=len(tokenized_datasets["train"]) // 16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,  
)

# Define the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()