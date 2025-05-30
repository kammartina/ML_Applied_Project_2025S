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

from train_pipeline import preprocess_and_tokenize, get_train_val_datasets, train, load_config
import torch
#import yaml
import os
import optuna
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import set_seed
set_seed(24)



def objective(trial, train_file, val_file, model_name, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, output_dir, logging_dir):
    # Suggest hyperparameters
    trial_num = trial.number
    trial_output_dir = os.path.join(output_dir, f"trial_{trial_num}")
    trial_logging_dir = os.path.join(logging_dir, f"trial_{trial_num}")
    

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 4, 8)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
    max_length = trial.suggest_int("max_length", max_tgt_length, 256)
    num_beams = trial.suggest_int("num_beams", 1, 4)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file, shuffle=True)
    tokenized_train = preprocess_and_tokenize(
        train_dataset, tokenizer, src_lang, tgt_lang,
        prefix, max_src_length, max_tgt_length, desc=f"Tokenizing train data"
    )
    tokenized_val = preprocess_and_tokenize(
        val_dataset, tokenizer, src_lang, tgt_lang,
        prefix, max_src_length, max_tgt_length, desc=f"Tokenizing val data"
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer, _, eval_results = train(
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
        early_stopping=True,
        early_stopping_patience=1,
        warmup_steps=warmup_steps,
        num_beams=num_beams,
        max_length=max_length
    )

    model_save_dir = os.path.join(trial_output_dir, "model")
    trainer.save_model(model_save_dir)  # Save the model for this trial
    bleu_score = eval_results.get("eval_bleu", 0)
    return bleu_score

  
def main():
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
   
    output_dir = config['output_dir']
    logging_dir = config['logging_dir']
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_file, val_file, model_name, src_lang, tgt_lang, prefix, max_src_length, max_tgt_length, output_dir, logging_dir), n_trials=5)
    print("Best hyperparameters:", study.best_params)
    
  
if __name__ == "__main__":
    main()