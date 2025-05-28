import os
import json
import csv

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def chunk_file(input_path, output_dir, output_prefix, chunk_size):
    """
    Splits a large file into smaller files with chunk_size lines each.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as infile:
        chunk_num = 0
        lines = []
        for i, line in enumerate(infile, 1):
            lines.append(line)
            if i % chunk_size == 0:
                out_path = os.path.join(output_dir, f"{output_prefix}_chunk{chunk_num}.jsonl")
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.writelines(lines)
                lines = []
                chunk_num += 1
        # Write any remaining lines
        if lines:
            out_path = os.path.join(output_dir, f"{output_prefix}_chunk{chunk_num}.jsonl")
            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines)

def jsonl_tsv(jsonl_path):
    """
    Converts a jsonl file to a tsv file.
    This function assumes the jsonl file contains objects with consistent keys.
    """
    base, _ = os.path.splitext(jsonl_path)
    tsv_path = base + '.tsv'

    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file, \
        open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        for line_num, line in enumerate(jsonl_file, 1):
            try:
                entry = json.loads(line)
                translation = entry.get("translation", {})
                en = translation.get("en", "").replace("\t", " ").strip()
                de = translation.get("de", "").replace("\t", " ").strip()

                # Skip lines that are missing either field
                if not en or not de:
                    print(f"Line {line_num}: missing 'en' or 'de', skipped.")
                    continue

                # Write exactly two columns
                tsv_file.write(f"{en}\t{de}\n")

            except json.JSONDecodeError:
                print(f"Line {line_num}: invalid JSON, skipped.")

    print(f"Saved TSV to '{tsv_path}'")

def main():
    #train_file = "Data/train.clean.jsonl"
    #val_file = "Data/val.clean.jsonl"
    #output_dir = "Data/chunk_files"

    #train_lines = count_lines(train_file)
    #val_lines = count_lines(val_file)
    #print(f"Total lines in train.jsonl: {train_lines}")
    #print(f"Total lines in val.jsonl: {val_lines}")

    #chunk_file(train_file, output_dir, "train", 16384)
    #chunk_file(val_file, output_dir, "val", 2048)
    jsonl_file = "Data/test.jsonl"
    jsonl_tsv(jsonl_file)


if __name__ == "__main__":
        main()