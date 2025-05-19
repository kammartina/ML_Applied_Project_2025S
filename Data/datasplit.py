import json
import zipfile
import os
import random
from sklearn.model_selection import train_test_split

# File path
file_path = "ted_aligned_en_de.json.zip"

# Extract and load data
def load_data_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        json_filename = os.path.splitext(os.path.basename(zip_path))[0]
        with z.open(json_filename) as f:
            data = json.load(f)
    return data

# Split data into train, validation, and test sets
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    return train_data, val_data, test_data

# Main function
def main():
    # Load data
    print("Loading data...")
    raw_data = load_data_from_zip(file_path)

    # Shuffle sentences within each talk
    for talk in raw_data.values():
        random.shuffle(talk)
    
    # Transform dictionary into a list of sentence pairs
    data = []
    for key, sentences in raw_data.items():
        for sentence in sentences:
            if "en" in sentence and "de" in sentence:
                data.append([sentence["en"], sentence["de"]])
    
    # Ensure data is a list of sentence pairs
    if not isinstance(data, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in data):
        raise ValueError("Data format is invalid. Expected a list of [source, target] sentence pairs.")
    
    
    # Split data
    print("Splitting data...")
    train_data, val_data, test_data = split_data(data)
    
    # Save splits to files
    print("Saving splits...")
    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open("val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    print("Data split and saved successfully!")

if __name__ == "__main__":
    main()