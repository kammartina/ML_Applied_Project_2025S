import yaml
import os
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import set_seed
import optuna
from sacrebleu import corpus_bleu
import datetime
from tqdm import tqdm
from functools import partial

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def get_train_val_datasets(train_file, val_file, fraction=0.01, max_train=800, max_valid=100, shuffle=False):
    """Load and return train and validation datasets (optionally shuffled and subsetted)."""
    print("Loading raw datasets...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
        split={
            "train": f"train[:{int(fraction * 100)}%]",
            "validation": f"validation[:{int(fraction * 100)}%]"
        }
    )

    print("Selecting subsets...")
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    if shuffle:
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
    if len(train_dataset) > max_train:
        train_dataset = train_dataset.select(range(max_train))
    if len(val_dataset) > max_valid:
        val_dataset = val_dataset.select(range(max_valid))

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    return train_dataset, val_dataset

def preprocess_and_tokenize(dataset, model_type, tokenizer, src_lang, tgt_lang, max_src_length, max_tgt_length, desc="Tokenizing dataset"):
    """Preprocess and tokenize a dataset."""
    def preprocess_function(examples):
        inputs, targets = [], []
        for item in examples["translation"]:
            src = item.get(src_lang)
            tgt = item.get(tgt_lang)
            if src and tgt:
                if model_type == "t5":
                    input_text = f"translate {src_lang} to {tgt_lang}: {src}"
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

#def objective(trial, model, tokenizer, tokenized_train, tokenized_val, data_collator):
    #try:
        # Sample hyperparameters
        #learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        #batch_size = trial.suggest_categorical("batch_size", [10, 20])
        #num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)

        # Log hyperparameters to WandB
        #wandb.config.update({
        # "learning_rate": learning_rate,
        #"batch_size": batch_size,
            #"num_train_epochs": num_train_epochs
    # })
def train(model, tokenizer, tokenized_train, tokenized_val, data_collator, learning_rate, weight_decay, batch_size, num_epoch, seed = 224):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"NMT_training_pipeline/trained_model",
        logging_dir=f"NMT_training_pipeline/logs",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epoch,
        logging_strategy="steps",
        logging_steps=40,
        report_to="tensorboard",
        predict_with_generate=True,
        seed=seed
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer) 
    )

        
    trainer.train()
    # Save the model
    trainer.save_model()
    # Evaluate and return BLEU score
    eval_results = trainer.evaluate()
    bleu_score = eval_results.get("eval_bleu", None)
    print(f"BLEU score: {bleu_score}")
    return bleu_score

  
def main():
    # Initialize WandB
    #wandb.init(project="ML for Translation Task", name="Model Training")
    summary_path = "NMT_training_pipeline/training_summary.tsv"
    config_path = "NMT_training_pipeline/configs/nllb_config.yml"
    print("Loading configurations:")


    config = load_config(config_path)
    set_seed(config["seed"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"])
    model_type = config["model_type"]
    src_lang_code = config.get["src_lang_code", ""]
    tgt_lang_code = config.get["src_lang_code", ""]
    src_lang = config['source_lang']
    tgt_lang = config['target_lang']
    max_src_length = config["max_source_length"]
    max_tgt_length = config["max_target_length"]
    train_file = config["train_file"]
    val_file = config["val_file"]

    if src_lang_code and tgt_lang_code:
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

    if model_type.lower().startswith("mbart"):
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    # Load and preprocess data
    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file)
    tokenized_train = preprocess_and_tokenize(train_dataset, model_type, tokenizer, src_lang, tgt_lang, max_src_length, max_tgt_length, desc="Tokenizing train dataset")
    tokenized_val = preprocess_and_tokenize(val_dataset, model_type, tokenizer, src_lang_code, tgt_lang_code, max_src_length, max_tgt_length, desc="Tokenizing validation dataset")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    batch_size = config["batch_size"]
    num_epoch = config["num_epoch"]

    #study = optuna.create_study(direction="maximize")  # Maximize BLEU score
    #study.optimize((partial(
        #objective,
        #model=model,
        #tokenizer=tokenizer,
        #tokenized_train=tokenized_train,
        #tokenized_val=tokenized_val,
        #data_collator=data_collator
    #)), n_trials=2)  # Run 2 trials

    # Print the best trial
    #print("Best trial:")
    #print(f" Value: {study.best_trial.value}")
    #print("  Params:")
    #for key, value in study.best_trial.params.items():
       # print(f"    {key}: {value}")
        #config[key] = value  
    bleu_score = train(
        model, 
        tokenizer, 
        tokenized_train, 
        tokenized_val, 
        data_collator, 
        learning_rate, 
        weight_decay, 
        batch_size, 
        num_epoch  
    )
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("model\ttime\tbleu_score\tlearning_rate\tbatch_size\tnum_epoch\tweight_decay\n")
        f.write(f"{model}\t{timestamp}\t{bleu_score}\t{learning_rate}\t{batch_size}\t{num_epoch}\t{weight_decay}\n")
    
    
    # Add date/time and BLEU score to config
    #config["last_trained"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #config["bleu_score"] = bleu_score

    #with open(config_path, "w") as file:
        #yaml.dump(config, file)

    #print(f"Updated config saved to {config_path}")
if __name__ == "__main__":
    main()