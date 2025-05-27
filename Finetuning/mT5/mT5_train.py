import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set this before importing torch

import torch
available_gpus = torch.cuda.device_count()
if available_gpus > 0:
    device = "cuda:0"
else:
    device = "cpu"

print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

import sys
sys.path.append('/home/mlt_ml2/ML_Applied_Project_2025S/NMT_training_pipeline')

from train import preprocess_and_tokenize, get_train_val_datasets, train, load_config
import torch
#import yaml
import os
import optuna
import glob
#from optuna.integration import HuggingFacePruningCallback
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import set_seed
#from sacrebleu import corpus_bleu
#import time
from functools import partial
set_seed(24)




def objective(trial, train_file, val_file, model_dir, model_type, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, idx):
    # Suggest hyperparameters
    trial_num = trial.number
    trial_output_dir = f"./mt5_results/chunk{idx}_trial{trial.number}"
    trial_logging_dir = f"./mt5_logs/chunk{idx}_trial_{trial_num}"
    

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 2, 6)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
    max_length = trial.suggest_int("max_length", max_tgt_length, 256)
    num_beams = trial.suggest_int("num_beams", 1, 4)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file, shuffle=True)
    tokenized_train = preprocess_and_tokenize(
        train_dataset, model_type, tokenizer, src_lang, tgt_lang,
        prefix, max_src_length, max_tgt_length, desc=f"Tokenizing train chunk {idx}"
    )
    tokenized_val = preprocess_and_tokenize(
        val_dataset, model_type, tokenizer, src_lang, tgt_lang,
        prefix, max_src_length, max_tgt_length, desc=f"Tokenizing val chunk {idx}"
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    #output_dir = f"{config['output_dir']}_chunk{idx}_trial{trial.number}"
    #logging_dir = f"{config['logging_dir']}_chunk{idx}_trial{trial.number}"


    trainer, bleu_score, eval_loss, train_loss = train(
        model, 
        tokenizer, 
        trial_output_dir, 
        trial_logging_dir, 
        tokenized_train, 
        tokenized_val, 
        data_collator, 
        learning_rate, 
        weight_decay, 
        batch_size, 
        num_epochs, 
        seed=224,
        trial=trial,
        warmup_steps=warmup_steps,
        num_beams=num_beams,
        max_length=max_length
    )
    model_save_dir = os.path.join(trial_output_dir, "model")
    trainer.save_model(model_save_dir)  # Save the model for this trial
    # Return the metric to maximize (or minimize)
    return bleu_score if bleu_score is not None else 0.0, eval_loss if eval_loss is not None else 0.0, train_loss if train_loss is not None else float("inf")

config_path = "NMT_training_pipeline/configs/mT5_config.yml"
print("Loading configurations:")


config = load_config(config_path)
set_seed(config["seed"])

# Load tokenizer and model
model_dir = config["model_name"]
model_type = config["model_type"]
src_lang_code = config.get("src_lang_code", "")
tgt_lang_code = config.get("tgt_lang_code", "")
src_lang = config['source_lang']
tgt_lang = config['target_lang']
prefix = config.get("prefix", "")
max_src_length = config["max_source_length"]
max_tgt_length = config["max_target_length"]
#train_file = config["train_file"]
#val_file = config["val_file"]
#output_dir = "./mt5_results"
#logging_dir = "./mt5_logs"

train_dir = config["train_dir"]
val_dir = config["val_dir"]
num_chunks = 2

train_chunks = sorted(glob.glob(os.path.join(train_dir, "train_chunk*.jsonl")))
val_chunks = sorted(glob.glob(os.path.join(val_dir, "val_chunk*.jsonl")))
# Select the chunk you want to use (e.g., the first one)
for idx, (train_file, val_file) in enumerate(zip(train_chunks, val_chunks)):
    print(f"\n=== Training on chunk {idx} ===")
    train_file = train_chunks[idx]
    val_file = val_chunks[idx]
    objective_partial = partial(
        objective,
        train_file=train_file,
        val_file=val_file,
        model_dir=model_dir,
        model_type=model_type,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prefix=prefix,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
        idx=idx
    )
#tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
#model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")

#train_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/train.clean.jsonl"
#val_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/val.clean.jsonl"
#model_type = "t5"
#src_lang = "en"
#target_lang = "de"
#prefix = "translate English to German: "
#max_src_length = 128
#max_tgt_length = 128
#data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
#train_dataset, val_dataset = get_train_val_datasets(train_file, val_file, max_train=16384, max_valid=2048, shuffle=False)
#tokenized_train = preprocess_and_tokenize(train_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing train dataset")
#tokenized_val = preprocess_and_tokenize(val_dataset, model_type, tokenizer, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, desc="Tokenizing validation dataset")



# Run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_partial, n_trials=5)
    print("Best trial for chunk {idx}:", study.best_trial)

# Calculate data size
    #train_size = len(tokenized_train)
    #val_size = len(tokenized_val)
    #num_trials = len(study.trials)
    #best_lr = study.best_trial.params["learning_rate"]

    log_num = 1
    while os.path.exists(f"mt5_optuna_log_{idx}_{log_num}.txt"):
        log_num += 1
    log_path = f"mt5_optuna_log_{idx}_{log_num}.txt"
    with open(log_path, "a") as f:
        f.write(f"Chunk index: {idx}\n")
        f.write(f"Train file: {train_file}\n")
        f.write(f"Val file: {val_file}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial params: {study.best_trial.params}\n")
        f.write(f"Best BLEU: {study.best_trial.value}\n")
        f.write("="*40 + "\n")
    print(f"Optuna results for chunk {idx} logged to {log_path}")





