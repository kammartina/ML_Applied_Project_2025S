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
from sacrebleu import corpus_bleu
import time
from functools import partial
import glob
import numpy as np

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

def preprocess_and_tokenize(dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing dataset"):
    """Preprocess and tokenize a dataset."""
    def preprocess_function(examples):
        inputs, targets = [], []
        for item in examples["translation"]:
            src = item.get(src_lang)
            tgt = item.get(tgt_lang)
            if src and tgt:
                if model_type == "t5":
                    input_text = f"{prefix}{src}"
                    inputs.append(input_text)
                    targets.append(tgt)
                else:
                    inputs.append(src)
                    targets.append(tgt)
        model_inputs = tokenizer(inputs, max_length=max_src_length, padding=True, truncation=True)
        labels = tokenizer(targets, max_length=max_tgt_length, padding=True, truncation=True)
        labels_ids = [[(token if token != tokenizer.pad_token_id else -100) for token in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels_ids
        return model_inputs

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc
    )


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU score for predictions."""
    preds, labels = eval_preds
    #vocab_size = getattr(tokenizer, "vocab_size", 32000)
    # Convert all tokens to Python int to avoid np.int64 issues
    #preds = [[int(token) for token in pred] for pred in preds]
    #labels = [[int(token) for token in label] for label in labels]
    #preds = [[int(token) if isinstance(token, (int, float)) and 0 <= int(token) < vocab_size else tokenizer.pad_token_id for token in pred] for pred in preds]
    #labels = [[int(token) if isinstance(token, (int, float)) and 0 <= int(token) < vocab_size else tokenizer.pad_token_id for token in label] for label in labels]
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    
    if hasattr(preds, "cpu"):
        preds = preds.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    # Ensure preds are integers only
    preds = np.array(preds).astype(np.int32)
    labels = np.array(labels).astype(np.int32)
    
    print("Sample preds:", preds[:2])
    print("Sample labels:", labels[:2])
    # ... rest of your code ...
    
    # Replace -100 in the labels as we can't decode them
    labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
    # Decode predictions and labels
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # Compute BLEU score
    bleu_score = corpus_bleu(decoded_preds, decoded_labels).score

    return {"bleu": bleu_score}


def train(model, tokenizer, output_dir, logging_dir, tokenized_train, tokenized_val, 
          data_collator, learning_rate, weight_decay, batch_size, num_epochs, seed=224, **kwargs):
    
    warmup_steps=kwargs.get("warmup_steps", 0)
    max_length=kwargs.get("max_length", None)
    num_beams=kwargs.get("num_beams", None)
    logging_steps=len(tokenized_train) // batch_size
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
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

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    #if trial is not None:
        #from optuna.integration.transformers import HuggingFacePruningCallback
        #callbacks.append(HuggingFacePruningCallback(trial, "eval_bleu"))
        
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
    bleu_score = eval_results.get("eval_bleu", None)
    eval_loss = eval_results.get("eval_loss", None)
    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    print(f"BLEU score: {bleu_score}")
    return trainer, bleu_score, eval_loss, train_loss

  
def main():
    # Initialize WandB
    #wandb.init(project="ML for Translation Task", name="Model Training")
    summary_path = "NMT_training_pipeline/training_summary_v2.tsv"
    config_path = "NMT_training_pipeline/configs/t5_config.yml"
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
    

    if src_lang_code and tgt_lang_code:
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

    if model_type.lower().startswith("mbart"):
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

       
    #train_dir = config["train_dir"]
    #val_dir = config["val_dir"]
    #num_chunks = 1

    #train_chunks = sorted(glob.glob(os.path.join(train_dir, "train_chunk*.jsonl")))[:num_chunks]
    #val_chunks = sorted(glob.glob(os.path.join(val_dir, "val_chunk*.jsonl")))[:num_chunks]

    #for idx, (train_file, val_file) in enumerate(zip(train_chunks, val_chunks)):
        #print(f"\n=== Training on chunk {idx+1} ===")
        # Load model and tokenizer from previous step
   
    
    # Load and preprocess data
    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file, max_train=800, max_valid=100, shuffle=True)
    tokenized_train = preprocess_and_tokenize(
        train_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc=f"Tokenizing train data"
    )
    tokenized_val = preprocess_and_tokenize(
        val_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc=f"Tokenizing val data"
    )

    # Load and preprocess data
    #train_dataset, val_dataset = get_train_val_datasets(train_file, val_file)
    #tokenized_train = preprocess_and_tokenize(train_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing train dataset")
    #tokenized_val = preprocess_and_tokenize(val_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing validation dataset")

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
    _, bleu_score, eval_loss, train_loss = train(
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
    #trainer.save_model(output_dir)
    #model_dir = output_dir
    #print(f"Model after chunk {idx} saved to {output_dir}")
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    training_time_str = f"{minutes:02d}:{seconds:02d}"
    
    
    row = "{:<40}{:<16}{:<16}{:<16}{:<13}{:<15}{:<12}{:<12}{:<12}{:<12}\n".format(
        config['model_name'],
        training_time_str,
        f"{train_loss:.2f}" if train_loss is not None else "NA",
        f"{eval_loss:.2f}" if eval_loss is not None else "NA",
        f"{bleu_score:.2f}" if bleu_score is not None else "NA",
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
            header = "{:<40}{:<16}{:<16}{:<16}{:<13}{:<15}{:<12}{:<12}{:<12}{:<12}\n".format(
                "model", "train_time", "train_loss", "eval_loss", "bleu_score", "learning_rate", "batch_size", "num_epochs", "weight_decay", "max_output_length"
                )
            f.write(header)
        f.write(row)
    print("Training summary written to:", summary_path)
    print("Training completed.")
    
    
  
if __name__ == "__main__":
    main()