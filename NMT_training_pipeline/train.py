import yaml
#import wandb
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import set_seed
import optuna
from sacrebleu import corpus_bleu
import math
from tqdm import tqdm
from functools import partial

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

def preprocess_function(examples, tokenizer, config, model_type):
    inputs, targets = [], []

    for item in tqdm(examples["translation"], desc="Tokenizing examples", total=len(examples["translation"])):
        src = item.get(config["source_lang"])
        tgt = item.get(config["target_lang"])
        if src and tgt:
            if model_type == "t5":
                input_text = f"translate {config['source_lang']} to {config['target_lang']}: {src}"
                inputs.append(input_text)
                targets.append(tgt)
            else:
                inputs.append(src)
                targets.append(tgt)
       
    model_inputs = tokenizer(inputs, max_length=config["max_source_length"], padding=True, truncation=True)
    labels = tokenizer(targets, max_length=config["max_target_length"], padding=True, truncation=True)
    # Setup the tokenizer for targets
    labels_ids = [[(token if token != tokenizer.pad_token_id else -100) for token in label] for label in labels["input_ids"]]
    
    model_inputs["labels"] = labels_ids
    return model_inputs

def load_and_preprocess_data(config, tokenizer, fraction=0.01, max_train = 800, max_valid = 100, max_test = 100, shuffle=False):
    """Load and preprocess the dataset."""
    # Load raw datasets with a progress bar
    print("Loading raw datasets...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": config["train_file"], "validation": config["val_file"]},
        split={
        "train": f"train[:{int(fraction * 100)}%]",
        "validation": f"validation[:{int(fraction * 100)}%]"
    }
    )

    # Select a subset of the data for training and validation
    print("Selecting subsets...")
    if shuffle:
        train_dataset = raw_datasets["train"].shuffle(seed=42)
        val_dataset = raw_datasets["validation"].shuffle(seed=42)
    if len(raw_datasets["train"]) > max_train:
        train_dataset = raw_datasets["train"].select(range(max_train))
    if len(raw_datasets["validation"]) > max_valid:
        val_dataset = raw_datasets["validation"].select(range(max_valid))
    #if len(dataset_dict["test"]) > max_test:
        #dataset_dict["test"] = dataset_dict["test"].select(range(max_test))
    #train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(int(len(raw_datasets["train"]) * fraction)))
    #val_dataset = raw_datasets["validation"].shuffle(seed=42).select(range(int(len(raw_datasets["validation"]) * fraction)))

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")

    # Tokenize datasets
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer, config, config["model_type"]),
        batched=True,
        remove_columns=train_dataset.column_names, desc ="Tokenizing train dataset"
    )
    # Tokenize validation dataset
    tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer, config, config["model_type"]),
        batched=True,
        remove_columns=val_dataset.column_names, desc="Tokenizing validation dataset"
    )

    return tokenized_train, tokenized_val


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


    print("Decoded Predictions:", decoded_preds[:5])  # Print first 5 predictions
    print("Decoded Labels:", decoded_labels[:5]) 

    # Compute BLEU score
    bleu_score = corpus_bleu(decoded_preds, decoded_labels).score

    return {"bleu": bleu_score}

def objective(trial, model, tokenizer, tokenized_train, tokenized_val, data_collator):
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16])
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)

    # Log hyperparameters to WandB
    #wandb.config.update({
       # "learning_rate": learning_rate,
       #"batch_size": batch_size,
        #"num_train_epochs": num_train_epochs
   # })

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./optuna_trial_{trial.number}",
        logging_dir=f"./logs/optuna_trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        predict_with_generate=True,
        seed=224
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

    for epoch in tqdm(range(num_train_epochs), desc="Training epochs"):     
        print(f"Epoch {epoch + 1}/{num_train_epochs}")
        # Train the model
        trainer.train()
    # Log the checkpoint directory to WandB as an artifact
    #artifact = wandb.Artifact("model-checkpoint", type="model")
    #artifact.add_dir(f"./optuna_trial_{trial.number}")  # Add the checkpoint directory dynamically
    #wandb.log_artifact(artifact)

def main():
    # Initialize WandB
    #wandb.init(project="ML for Translation Task", name="Model Training")
    config_path = "/home/mlt_ml2/ML_Applied_Project_2025S/NMT_training_pipeline/configs/mbart50_config.yml"
    print("Loading configurations:")


    config = load_config(config_path)
    set_seed(config["seed"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"])

    tokenizer.src_lang = config["src_lang_code"]
    tokenizer.tgt_lang = config["tgt_lang_code"]
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(config["tgt_lang_code"])

    # Load and preprocess data
    tokenized_train, tokenized_val = load_and_preprocess_data(config, tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    study = optuna.create_study(direction="maximize")  # Maximize BLEU score
    study.optimize((partial(
        objective,
        model=model,
        tokenizer=tokenizer,
        tokenized_train=tokenized_train,
        tokenized_val=tokenized_val,
        data_collator=data_collator
    )), n_trials=2)  # Run 5 trials

    # Print the best trial
    print("Best trial:")
    print(f" Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        config[key] = value    

    with open(config_path, "w") as file:
        yaml.dump(config, file)

    print(f"Updated config saved to {config_path}")
if __name__ == "__main__":
    main()