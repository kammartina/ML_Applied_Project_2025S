"""
This script provides code for using the comparison tool "compare-mt"

This was done in Google Colab.
"""

!pip install transformers
!pip install sentencepiece

!pip install compare-mt

import os
import torch
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# Define checkpoints
model_paths = {
    "t5_orig_en_de": "/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_en-de_finetuned(32K)/checkpoint-6000",
    "t5_cleaned_en_de": "/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_en-de_finetuned(32K)_cleaned_dataset/checkpoint-8000",
    "t5_cleaned_en_fr": "/content/drive/MyDrive/ML2_Bublin_Project/Martina_T5/t5_base_en-de_finetuned(32K)_cleaned_dataset/checkpoint-8000",
    "nllb_orig_en_de": "/content/drive/MyDrive/ML2_Bublin_Project/Sandra_NLLB/nllb_en-de_finetuned(32K)/checkpoint-8000",
    "nllb_cleaned_en_de": "/content/drive/MyDrive/ML2_Bublin_Project/Sandra_NLLB/nllb_en-de_finetuned(32K)_cleaned_dataset/checkpoint-10000",
    "nllb_cleaned_en_fr": "/content/drive/MyDrive/ML2_Bublin_Project/Sandra_NLLB/nllb_en-fr_finetuned(32K)_cleaned_dataset/checkpoint-10000"
}

# Paths to source and reference files
source_file = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/original_text_en.txt"
reference_de = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_de.txt"
reference_fr = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_fr.txt"

# Output files for each system
output_dir_en = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/compare-mt_outputs_en_de"
output_dir_fr = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/compare-mt_outputs_en_fr"
os.makedirs(output_dir_en, exist_ok=True)
os.makedirs(output_dir_fr, exist_ok=True)

from pprint import pprint
with open(source_file, 'r', encoding='utf-8') as src_f:
    src_lines = src_f.readlines()
    print(len(src_lines))
    pprint(src_lines[:10])

check_dir = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_fr.txt"
with open(check_dir, 'r', encoding='utf-8') as src_f:
    src_lines = src_f.readlines()
    print(len(src_lines))
    pprint(src_lines[:10])

## German and French don't have the correct encoding
# Re-encode the file from ISO-8859-1 to UTF-8
input_path = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_fr.txt"
output_path = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_fr.txt"

with open(input_path, 'r', encoding='ISO-8859-1') as infile:
    content = infile.read()

with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.write(content)

print("File re-encoded to UTF-8 and saved as", output_path)

from pprint import pprint
check_dir = "/content/drive/MyDrive/ML2_Bublin_Project/compare-mt/reference_translation_fr.txt"
with open(check_dir, 'r', encoding='utf-8') as src_f:
    src_lines = src_f.readlines()
    print(len(src_lines))
    pprint(src_lines[:10])


"""# Generating translations"""
# Translation function
def translate_and_save(model_path, source_file, output_file, prefix="", src_lang=None, tgt_lang=None, batch_size=8, max_length=500):
    ## t5
    if "t5" in model_path.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    ## nllb
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Pass src_lang and tgt_lang to the pipeline for NLLB models
    if "nllb" in model_path.lower():
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, max_length=max_length, src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1, max_length=max_length)


    with open(source_file, 'r', encoding='utf-8') as src_f:
        src_lines = src_f.readlines()

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i in range(0, len(src_lines), batch_size):
            batch = src_lines[i:i+batch_size]
            if "t5" in model_path.lower() and prefix:
                batch = [prefix + " " + line.strip() for line in batch]
            #elif "nllb" in model_path.lower() and prefix:
            #    batch = [prefix + " " + line.strip() for line in batch]
            translations = translator(batch)
            for t in translations:
                out_f.write(t['translation_text'].strip() + '\n')

# Translate with all models
translate_and_save(model_paths["t5_orig_en_de"], source_file, f"{output_dir_en}/t5_orig_en_de.txt", prefix="translate English to German: ")
print("1. Translated 't5_orig_en_de' completed.\n")

translate_and_save(model_paths["t5_cleaned_en_de"], source_file, f"{output_dir_en}/t5_cleaned_en_de.txt", prefix="translate English to German: ")
print("2. Translated 't5_cleaned_en_de' completed.\n")

translate_and_save(model_paths["t5_cleaned_en_fr"], source_file, f"{output_dir_en}/t5_cleaned_en_fr.txt", prefix="translate English to German: ")
print("3. Translated 't5_cleaned_en_fr' completed.\n")

translate_and_save(model_paths["nllb_orig_en_de"], source_file, f"{output_dir_en}/nllb_orig_en_de.txt", src_lang="eng_Latn", tgt_lang="deu_Latn")
print("4. Translated 'nllb_orig_en_de' completed.\n")

translate_and_save(model_paths["nllb_cleaned_en_de"], source_file, f"{output_dir_en}/nllb_cleaned_en_de.txt", src_lang="eng_Latn", tgt_lang="deu_Latn")
print("5. Translated 'nllb_cleaned_en_de' completed.\n")

translate_and_save(model_paths["nllb_cleaned_en_fr"], source_file, f"{output_dir_en}/nllb_cleaned_en_fr.txt", src_lang="eng_Latn", tgt_lang="deu_Latn")
print("6. Translated 'nllb_cleaned_en_fr' completed.")

translate_and_save(model_paths["t5_cleaned_en_fr"], source_file, f"{output_dir_fr}/t5_cleaned_en_fr.txt", prefix="translate English to French: ")
print("1. Translated 't5_cleaned_en_fr' completed.\n")

translate_and_save(model_paths["nllb_cleaned_en_fr"], source_file, f"{output_dir_fr}/nllb_cleaned_en_fr.txt", src_lang="eng_Latn", tgt_lang="fra_Latn")
print("2. Translated 'nllb_cleaned_en_fr' completed.")


"""# Comparing EN to DE (all models)"""
# compare if all the files have the same length
!wc -l reference_translation_de.txt t5_orig_en_de.txt t5_cleaned_en_de.txt t5_cleaned_en_fr.txt nllb_orig_en_de.txt nllb_cleaned_en_de.txt nllb_cleaned_en_fr.txt

!compare-mt --output_directory output/ reference_translation_de.txt t5_orig_en_de.txt t5_cleaned_en_de.txt t5_cleaned_en_fr.txt nllb_orig_en_de.txt nllb_cleaned_en_de.txt nllb_cleaned_en_fr.txt --compare_scores score_type=sacrebleu,bootstrap=1000,prob_thresh=0.05 --decimals 2 --fig_size 10x5


"""# Comparing EN to FR (#3 and #6)"""
# compare if all the files have the same length
!wc -l reference_translation_fr.txt t5_cleaned_en_fr.txt nllb_cleaned_en_fr.txt

## upravit
!compare-mt --output_directory output/ reference_translation_fr.txt t5_cleaned_en_fr.txt nllb_cleaned_en_fr.txt --compare_scores score_type=sacrebleu,bootstrap=1000,prob_thresh=0.05 --decimals 2 --fig_size 10x5