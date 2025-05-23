########## T5-base model fine-tuning for ENGLISH to FRENCH Translation ##########
# This script was run in the Google Colab environment.
"""
!pip install datasets evaluate transformers[sentencepiece]
!pip install sacrebleu
"""

# Import necessary libraries
#import torch
import os
import json
import random
import evaluate
import numpy as np

from pprint import pprint
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

"""
# Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
"""

# Define paths for each split
# the splits were uploaded into the Colab
train_path = "en-fr_train.clean.jsonl"
val_path = "en-fr_val.clean.jsonl"
test_path = "en-fr_test.clean.jsonl"

# --------------------------
# Load the dataset splits
# --------------------------
train_data = load_dataset("json", data_files=train_path)
val_data = load_dataset("json", data_files=val_path)
test_data = load_dataset("json", data_files=test_path)


# --------------------------
# Create DatasetDict
# --------------------------
dataset_dict = DatasetDict({
    'train': train_data["train"],
    'validation': val_data["train"],
    'test': test_data["train"]
})

# Print the dataset dictionary
print("Dataset Dictionary:")
pprint(dataset_dict)

# Print the first few examples from the training set
print("\nSample of the training set:")
pprint(dataset_dict["train"][0])
pprint(dataset_dict["train"][1])
pprint(dataset_dict["train"][2])


# --------------------------
# Subsample the Dataset (Before Tokenization)
# --------------------------
max_train = 32000  # batch size is 16
max_valid = 512
max_test = 512

# we don‘t have to shuffle here, because the Seq2SeqDataCollator will take care of that
if len(dataset_dict["train"]) > max_train:
    dataset_dict["train"] = dataset_dict["train"].select(range(max_train))
if len(dataset_dict["validation"]) > max_valid:
    dataset_dict["validation"] = dataset_dict["validation"].select(range(max_valid))
if len(dataset_dict["test"]) > max_test:
    dataset_dict["test"] = dataset_dict["test"].select(range(max_test))

print("\nAfter subsampling:")
pprint({k: len(v) for k, v in dataset_dict.items()})

# --------------------------
# Model and Tokenizer Setup
# --------------------------

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_en-de_finetuned(32K)_cleaned_dataset/checkpoint-8000")

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")


# --------------------------
# Define the Preprocessing Function Using a Natural Language Prefix
# --------------------------
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

max_length=250

def preprocess_function(examples):
    # this goes to the encoder
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    # this goes to the decoder
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, padding=False, truncation=True)

    # this explicitly re-tokenizes the "targets" to ensure that the labels are correctly aligned with the input_ids
    labels = tokenizer(targets, max_length=max_length, padding=False, truncation=True)

    # this overwrites whatever was in "model_inputs["labels"]" with the freshly computed target token IDs"
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Map the preprocessing function to the dataset
tokenized_dataset = dataset_dict.map(preprocess_function, batched=True).remove_columns(["translation"])

# Print the first few tokenized examples
print("\nSample of the tokenized training set:")
pprint(tokenized_dataset["train"][0])
pprint(tokenized_dataset["train"][1])

# after map(…)
print("\n...after mapping:")
print(tokenized_dataset["train"][0]["input_ids"][:10])
print(tokenizer.decode(tokenized_dataset["train"][0]["input_ids"][:20]))

# --------------------------
# Evaluation Metrics: SACREBLEU
# --------------------------
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    ### Ensure the predictions are within the vocabulary size
    # It clips (limits) the values in the preds array to be within the valid range of token IDs for the tokenizer's vocabulary.
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# --------------------------
# Training Arguments and Trainer Setup
# --------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join("/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_en-fr_finetuned(32K)_cleaned_dataset"),  # Where to store model checkpoints and outputs
    num_train_epochs=4,                  # Total number of training epochs
    per_device_train_batch_size=16,      # Batch size per device during training
    per_device_eval_batch_size=16,       # Batch size per device during evaluation
    eval_strategy="epoch",               # Evaluate at the end of every epoch
    save_strategy="epoch",               # Save checkpoint after each epoch
    #logging_steps=100,                  # Log training metrics every 100 steps
    learning_rate=2e-5,                  # Learning rate; you can experiment with different values
    weight_decay=0.01,                   # Weight decay for optimization
    fp16=True,                           # Enable mixed-precision training if you have a compatible GPU
    report_to="none",                    # Disables W&B logging
    load_best_model_at_end=True,         # Load the best model at the end of training
    predict_with_generate=True,          # Use generate() for evaluation to compute metrics like BLEU if desired
    generation_max_length=250,           # Maximum length of generated sequences
    generation_num_beams=5,              # Number of beams for beam search during generation
)

# Initialize the data collator for dynamic padding during training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,                                    # My T5 model loaded and adjusted earlier
    args=training_args,                             # Training arguments defined above
    train_dataset=tokenized_dataset["train"],       # Tokenized training dataset
    eval_dataset=tokenized_dataset["validation"],   # Tokenized validation dataset
    tokenizer=tokenizer,                            # The tokenizer to use (with special tokens added)
    data_collator=data_collator,                    # Collator to handle batching and padding
    compute_metrics=compute_metrics
)

# --------------------------
# Train the Model
# --------------------------
trainer.train()

# --------------------------
# Evaluate on Validation Set
# --------------------------
val_results = trainer.predict(tokenized_dataset['validation'])
print(val_results)

# --------------------------
# Evaluate on Test Set
# --------------------------
#model_path = "/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_finetuned(32K)/checkpoint-6000"
#model_finetuned = .from_pretrained(model_path)

# BEARBEITEN
"""
trainer_test = Trainer(
    model=model_finetuned, # we give our model, our arguments and out dataset to the class Trainer
    args=training_args, # these are our parameters we set
    eval_dataset=tokenized_dataset['test'], # change to test when you do your final evaluation!
    processing_class=tokenizer,
    data_collator=data_collator, # processing method
    compute_metrics=compute_metrics # how the evaluation metrics is supposed to be computed
)
"""
# test_results = trainer_test.evaluate(eval_dataset=tokenized_dataset["test"])
# print("Test results:", test_results)

# --------------------------
# Save the Fine-Tuned Model
# --------------------------
#output_finetuned = os.path.join("/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_finetuned(32K)/epoch_3")
#trainer.save_model("/content/t5_base_translation/checkpoint-6000") # choose the checkpoint