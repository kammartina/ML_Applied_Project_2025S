# Use a pipeline as a high-level helper
#from transformers import pipeline
#from tqdm import tqdm

#translator = pipeline("translation", model="DunnBC22/mbart-large-50-English_German_Translation", device=0)
#source_lang = "en_XX"
#target_lang = "de_DE"
#source_text = "People are more likely to support the use"
#translation = translator(source_text, src_lang=source_lang, tgt_lang=target_lang)
#print(translation[0]['translation_text'])

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load the TSV file
tsv_file = "/home/mlt_ml2/ML_Applied_Project_2025S/datasplit_json/train.clean.tsv"
data = pd.read_csv(tsv_file, sep="\t", names=["en", "de"])  # Ensure the file is tab-separated

# Check the first few rows to understand the structure
print(data.head())

# Assuming the source text is in a column named "source_text"
source_texts = data["en"][:10].tolist()

# Initialize the translation pipeline
translator = pipeline("translation", model="DunnBC22/mbart-large-50-English_German_Translation", device=0)
source_lang = "en_XX"
target_lang = "de_DE"

# Translate each sentence
translations = []
for text in tqdm(source_texts, desc="Translating"):
    translation = translator(text, src_lang=source_lang, tgt_lang=target_lang)
    print(translation[0]['translation_text'])  # Access the first dictionary directly
    translations.append(translation[0]['translation_text'])  # Append the translation to the list


# Add translations to the DataFrame
#data["translation"] = translations

# Save the results to a new TSV file
#output_file = "/home/mlt_ml2/ML_Applied_Project_2025S/datasplit_json/train.clean.tsv"
#data.to_csv(output_file, sep="\t", index=False)
#print(f"Translations saved to {output_file}")