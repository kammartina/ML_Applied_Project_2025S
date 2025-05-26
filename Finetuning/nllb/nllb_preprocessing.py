
import os
import json
import zipfile
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer

# Paths
zip_path = "ted_aligned_en_de.json.zip"
json_path = "ted_aligned_en_de.json"
RAW_SPLIT_PATH = "huggingface_dataset_split"
TOKENIZED_PATH = RAW_SPLIT_PATH + "_tokenized"

# Extract ZIP and Load JSON
def extract_and_load_json(zip_path: str, json_path: str) -> dict:
    if not os.path.exists(json_path):
        print("Extracting JSON from ZIP archive.")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(json_path))
    else:
        print("JSON already extracted.")
    print("Loading JSON into memory.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Convert to Hugging Face Dataset
def convert_to_huggingface_dataset(raw_data: dict, src_lang="en", tgt_lang="de") -> Dataset:
    print("Converting raw JSON to Hugging Face Dataset.")
    src_texts, tgt_texts = [], []
    for segments in raw_data.values():
        for pair in segments:
            src = pair.get(src_lang, "").strip()
            tgt = pair.get(tgt_lang, "").strip()
            if src and tgt:
                src_texts.append(src)
                tgt_texts.append(tgt)
    return Dataset.from_dict({src_lang: src_texts, tgt_lang: tgt_texts})


# Split Dataset 
def split_dataset(dataset: Dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) -> DatasetDict:
    print("Splitting dataset...")
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    dataset = dataset.shuffle(seed=42)
    train_test_split_result = dataset.train_test_split(test_size=(1 - train_ratio), seed=42)
    test_val_split = train_test_split_result["test"].train_test_split(
        test_size=(test_ratio / (val_ratio + test_ratio)), seed=42
    )

    return DatasetDict({
        "train": train_test_split_result["train"],
        "validation": test_val_split["train"],
        "test": test_val_split["test"],
    })

# Tokenization
def tokenize_dataset(dataset_splits: DatasetDict, model_name: str = "facebook/nllb-200-distilled-600M") -> DatasetDict:
    print("Tokenizing dataset splits")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = "eng_Latn"
    tokenizer.tgt_lang = "deu_Latn"

    def tokenize_batch(batch):
        return tokenizer(
            batch["en"],
            text_target=batch["de"],
            padding=True,
            truncation=True
        )

    return dataset_splits.map(tokenize_batch, batched=True, remove_columns=["en", "de"])

# Save dataset splits to disk
def save_dataset_splits(dataset_splits: DatasetDict, save_path: str):
    if os.path.exists(save_path):
        print("Split dataset already saved.")
    else:
        print("Saving split dataset to disk.")
        dataset_splits.save_to_disk(save_path)


# Save JSON copies of the splits
def save_splits_to_json(dataset_splits: DatasetDict):
    print("Saving JSON files for each split")
    for split_name in ["train", "validation", "test"]:
        filename = f"{split_name}.json"
        data = list(zip(dataset_splits[split_name]["en"], dataset_splits[split_name]["de"]))
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# Full Pipeline
def load_or_process_dataset(zip_path: str, json_path: str, raw_path: str, tokenized_path: str) -> DatasetDict:
    if os.path.exists(tokenized_path):
        print("Loading pre-tokenized dataset.")
        return load_from_disk(tokenized_path)

    raw_data = extract_and_load_json(zip_path, json_path)
    dataset = convert_to_huggingface_dataset(raw_data)
    dataset_splits = split_dataset(dataset)
    save_dataset_splits(dataset_splits, raw_path)
    save_splits_to_json(dataset_splits)

    tokenized_dataset = tokenize_dataset(dataset_splits)
    print("Saving tokenized dataset to disk.")
    tokenized_dataset.save_to_disk(tokenized_path)

    return tokenized_dataset


if __name__ == "__main__":
    dataset_splits = load_or_process_dataset(zip_path, json_path, RAW_SPLIT_PATH, TOKENIZED_PATH)
    print(dataset_splits)
    print("Preprocessing and tokenization complete.")