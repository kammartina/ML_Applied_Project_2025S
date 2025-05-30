import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set this before importing torch

import torch
available_gpus = torch.cuda.device_count()
if available_gpus > 0:
    device = "cuda:0"
else:
    device = "cpu"


import yaml
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import set_seed
import evaluate
metric = evaluate.load("sacrebleu")
import time
from functools import partial
#import glob
import numpy as np
import math

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def get_train_val_datasets(train_file, val_file, max_train=None, max_valid=None, shuffle=False):
    """Load and return train and validation datasets (optionally shuffled and subsetted)."""
    print("Loading raw datasets...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
        split={
            "train": "train[:8000]",
            "validation": "validation[:1000]"
        }
    )

    print("Selecting subsets...")
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    if shuffle:
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
    if max_train is not None and len(train_dataset) > max_train:
        train_dataset = train_dataset.select(range(max_train))
    if max_valid is not None and len(val_dataset) > max_valid:
        val_dataset = val_dataset.select(range(max_valid))

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    return train_dataset, val_dataset

def preprocess_and_tokenize(dataset, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing dataset"):
    """Preprocess and tokenize a dataset."""
    def preprocess_function(examples):
        inputs = [ex[src_lang] for ex in examples["translation"]]
        targets = [ex[tgt_lang] for ex in examples["translation"]]
        if prefix:
            inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_src_length, padding=True, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_tgt_length, padding=True, truncation=True)
    
        if tokenizer.pad_token_id is not None:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc
    )


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    #show the number of tokens that need to be clipped
    num_clipped = np.sum(preds > tokenizer.vocab_size - 1)
    print(f"Number of tokens to be clipped: {num_clipped}")
    
    # Ensure the predictions are within the vocabulary size by clipping the values in the preds array to be within the valid range of token IDs for the tokenizer's vocabulary.
    max_token_id = max(tokenizer.get_vocab().values())
    preds = np.clip(preds, 0, max_token_id)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



def train(model, tokenizer, output_dir, logging_dir, tokenized_train, tokenized_val, 
          data_collator, learning_rate, weight_decay, batch_size, num_epochs, seed=224, callbacks=None, early_stopping=False, early_stopping_patience=1, **kwargs):
    
    warmup_steps=kwargs.get("warmup_steps", 0)
    max_length=kwargs.get("max_length", None)
    num_beams=kwargs.get("num_beams", None)
    logging_steps=len(tokenized_train) // batch_size

    if early_stopping:
        save_strategy = "epoch"
        save_total_limit = early_stopping_patience + 1  # Save the best model and a few checkpoints
        load_best_model_at_end = True
        metric_for_best_model = "eval_bleu"
        greater_is_better = True
        if callbacks is None:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    else:
        save_strategy = "no"
        save_total_limit = 1 # dummy assignment, No need to save intermediate models
        load_best_model_at_end = False
        metric_for_best_model = None
        greater_is_better = None
        if callbacks is None:
            callbacks = []
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_dir=logging_dir,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        **({"save_total_limit": save_total_limit} if save_strategy != "no" else {}),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        load_best_model_at_end=load_best_model_at_end,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_strategy="steps",
        logging_steps=logging_steps,
        report_to="tensorboard",
        predict_with_generate=True,
        generation_max_length=max_length, 
        generation_num_beams=num_beams, 
        seed=seed
    )

    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=callbacks
    )

        
    train_result = trainer.train()
    # Save the model
    trainer.save_model()
    # Evaluate and return BLEU score
    eval_results = trainer.evaluate()
    
    return trainer, train_result, eval_results

  
def main():
    summary_path = "NMT_training_pipeline/training_summary_v3.tsv"
    config_path = "NMT_training_pipeline/configs/mT5_config.yml"
    print("Loading configurations:")


    config = load_config(config_path)
    set_seed(config["seed"])

    # Load tokenizer and model
    model_name = config["model_name"]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_type = config["model_type"]
    src_lang_code = config.get("src_lang_code", "")
    tgt_lang_code = config.get("tgt_lang_code", "")
    src_lang = config['source_lang']
    tgt_lang = config['target_lang']
    prefix = config.get("prefix", "")
    max_src_length = config["max_source_length"]
    max_tgt_length = config["max_target_length"]
    train_file=config["train_file"]
    val_file=config["val_file"]
    
    if model_type == "t5":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if src_lang_code and tgt_lang_code:
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

    if model_type.lower().startswith("mbart"):
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
   
    
    # Load and preprocess data
    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file, max_train=800, max_valid=100, shuffle=True)
    tokenized_train = preprocess_and_tokenize(
        train_dataset, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc=f"Tokenizing train data"
    )
    tokenized_val = preprocess_and_tokenize(
        val_dataset, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc=f"Tokenizing val data"
    )

   
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    batch_size = int(config["batch_size"])
    num_epochs = int(config["num_train_epochs"])
    #num_beams = int(config.get("num_beams", None))
    max_output_length = int(config.get("max_target_length", 128))
    #warmup_steps = (len(tokenized_train) // batch_size) * num_epochs // 20  # 10% of training steps
    output_dir = config['output_dir']
    logging_dir = config['logging_dir']

    # Train the model
    start_time = time.time()
    _, train_result, eval_results = train(
        model, 
        tokenizer, 
        output_dir, 
        logging_dir, 
        tokenized_train, 
        tokenized_val, 
        data_collator, 
        learning_rate, 
        weight_decay, 
        batch_size, 
        num_epochs, 
        seed = 224,
        #warmup_steps=warmup_steps,
        #num_beams=num_beams,
        max_length=max_output_length
    )
    
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    training_time_str = f"{minutes:02d}:{seconds:02d}"

    bleu_score = eval_results.get("eval_bleu", None)
    eval_loss = eval_results.get("eval_loss", None)
    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    gen_len = eval_results.get("eval_gen_len", None)
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    
    row = "{:<40}{:<16}{:<16}{:<16}{:<13}{:<13}{:<13}{:<15}{:<12}{:<12}{:<14}{:<12}\n".format(
        config['model_name'],
        training_time_str,
        f"{train_loss:.2f}" if train_loss is not None else "NA",
        f"{eval_loss:.2f}" if eval_loss is not None else "NA",
        f"{perplexity:.2f}" if perplexity is not None else "NA",
        f"{bleu_score:.2f}" if bleu_score is not None else "NA",
        f"{gen_len:.2f}" if gen_len is not None else "NA",
        learning_rate,
        batch_size,
        num_epochs,
        weight_decay,
        #warmup_steps,
        max_output_length,
        #num_beams
    )
    
    
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            header = "{:<40}{:<16}{:<16}{:<16}{:<13}{:<13}{:<13}{:<15}{:<12}{:<12}{:<14}{:<12}\n".format(
                "model", "train_time", "train_loss", "eval_loss", "perplexity", "bleu_score", "gen_len", "learning_rate", "batch_size", "num_epochs", "weight_decay", "max_output_length"
                )
            f.write(header)
        f.write(row)
    print("Training summary written to:", summary_path)
    print("Training completed.")
    
    
  
if __name__ == "__main__":
    main()