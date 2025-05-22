import yaml
import os
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import set_seed
from sacrebleu import corpus_bleu
import time
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
def train(model, tokenizer, output_dir, logging_dir, tokenized_train, tokenized_val, data_collator, learning_rate, weight_decay, batch_size, num_epochs, seed = 224):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
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

        
    train_result = trainer.train()
    # Save the model
    trainer.save_model()
    # Evaluate and return BLEU score
    eval_results = trainer.evaluate()
    bleu_score = eval_results.get("eval_bleu", None)
    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    print(f"BLEU score: {bleu_score}")
    return bleu_score, train_loss

  
def main():
    # Initialize WandB
    #wandb.init(project="ML for Translation Task", name="Model Training")
    summary_path = "NMT_training_pipeline/training_summary.tsv"
    config_path = "NMT_training_pipeline/configs/mbart50_config.yml"
    print("Loading configurations:")


    config = load_config(config_path)
    set_seed(config["seed"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    model_type = config["model_type"]
    src_lang_code = config.get("src_lang_code", "")
    tgt_lang_code = config.get("tgt_lang_code", "")
    src_lang = config['source_lang']
    tgt_lang = config['target_lang']
    max_src_length = config["max_source_length"]
    max_tgt_length = config["max_target_length"]
    train_file = config["train_file"]
    val_file = config["val_file"]
    output_dir = config["output_dir"]
    logging_dir = config["logging_dir"]

    if src_lang_code and tgt_lang_code:
        tokenizer.src_lang = src_lang_code
        tokenizer.tgt_lang = tgt_lang_code

    if model_type.lower().startswith("mbart"):
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    # Load and preprocess data
    train_dataset, val_dataset = get_train_val_datasets(train_file, val_file)
    tokenized_train = preprocess_and_tokenize(train_dataset, model_type, tokenizer, src_lang, tgt_lang, max_src_length, max_tgt_length, desc="Tokenizing train dataset")
    tokenized_val = preprocess_and_tokenize(val_dataset, model_type, tokenizer, src_lang, tgt_lang, max_src_length, max_tgt_length, desc="Tokenizing validation dataset")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    batch_size = int(config["batch_size"])
    num_epochs = int(config["num_train_epochs"])

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
    start_time = time.time()
    bleu_score, train_loss = train(
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
        seed = 224 
    )
    
    training_time = time.time() - start_time
    minutes, seconds = divmod(int(training_time), 60)
    training_time_str = f"{minutes:02d}:{seconds:02d}"
    
    
    row = "{:<40}{:<16}{:<16}{:<13}{:<15}{:<12}{:<12}{:<12}\n".format(
        config['model_name'],
        training_time_str,
        f"{train_loss:.2f}" if train_loss is not None else "NA",
        f"{bleu_score:.2f}" if bleu_score is not None else "NA",
        learning_rate,
        batch_size,
        num_epochs,
        weight_decay
    )
    
    
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8") as f:
        if write_header:
            header = "{:<40}{:<16}{:<16}{:<13}{:<15}{:<12}{:<12}{:<12}\n".format(
                "model", "train_time", "train_loss","bleu_score", "learning_rate", "batch_size", "num_epochs", "weight_decay"
                )
            f.write(header)
        f.write(row)
    
    
  
if __name__ == "__main__":
    main()