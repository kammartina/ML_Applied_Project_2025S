from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate
from nllb_data_loader import load_tokenized_dataset

print("Loading dataset...")
# Load tokenized + split dataset
train_dataset, val_dataset = load_tokenized_dataset()

# Load pre-trained model + tokenizer
model_name = "/home/mlt_ml2/ML_Applied_Project_2025S/Finetuning/nllb/nllb_results_run3/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# NLLB-specific language codes
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "deu_Latn"

# Evaluation metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf") # Character n-gram F-score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    return {
        "sacrebleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["score"],
        "chrf": chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    }

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/mlt_ml2/ML_Applied_Project_2025S/Finetuning/nllb/nllb_results_run4",
    eval_strategy="epoch", 
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5, #smaller to aboid overfitting
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,  # if on supported GPU
    logging_dir="/home/mlt_ml2/ML_Applied_Project_2025S/Finetuning/nllb/nllb_results_run4/logs",
    logging_steps=100,
    save_total_limit=5,
    remove_unused_columns=False, # Keep "en" and "de" columns
    disable_tqdm=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    print("Saving model and tokenizer...")
    save_dir = "/home/mlt_ml2/ML_Applied_Project_2025S/Finetuning/nllb/nllb_results_run4"
    trainer.save_model(save_dir)
    trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Training complete.")
    print("Model and tokenizer saved.")
