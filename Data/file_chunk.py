import os

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


train_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/train.clean.jsonl"
val_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/val.clean.jsonl"
output_dir = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/chunk_files"

train_lines = count_lines(train_file)
val_lines = count_lines(val_file)
print(f"Total lines in train.jsonl: {train_lines}")
print(f"Total lines in val.jsonl: {val_lines}")

chunk_file(train_file, output_dir, "train", 16384)
chunk_file(val_file, output_dir, "val", 2048)