import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import sacrebleu


def extract_source_reference(tsv_file, source_lang, target_lang, num_sentences=10):
    """
    Extract source sentences from a TSV file.

    Args:
        tsv_file (str): Path to the TSV file with source and target columns.
        num_sentences (int): Number of sentences to extract.

    Returns:
        list: List of source sentences and target sentences as references for evaluation.
    """
    # Load the TSV file
    parallel_data = pd.read_csv(tsv_file, sep="\t", names=[f"{source_lang}", f"{target_lang}"])
    # Check if the file is empty
    if parallel_data.empty:
        raise ValueError("The TSV file is empty or not formatted correctly.")
    # Check if the specified columns exist
    if f"{source_lang}" not in parallel_data.columns or f"{target_lang}" not in parallel_data.columns:
        raise ValueError(f"The TSV file must contain columns named '{source_lang}' and '{target_lang}'.")
    # Check if the number of sentences requested is greater than the available data
    if num_sentences > len(parallel_data):
        raise ValueError(f"Requested number of sentences ({num_sentences}) exceeds available data ({len(parallel_data)}).")
    #print(parallel_data.head())

    # Select source texts
    source_texts = parallel_data[source_lang][:num_sentences].tolist()
    references = parallel_data[target_lang][:num_sentences].tolist()
    return source_texts, references

def translate_sentences(
    source_texts, 
    model_name, 
    model_type, # "t5", "mbart", "marian", etc.
    source_lang_code,
    target_lang_code,
    device=0
):
    """
    Translate sentences from a TSV file using a specified Hugging Face translation model.

    Args:
        model_name (str): Hugging Face model name or path.
        source_lang (str): Source language code for the model.
        target_lang (str): Target language code for the model.
        num_sentences (int): Number of sentences to translate.
        device (int): Device to run the model on (0 for GPU, -1 for CPU).

    Returns:
        list: List of translated sentences.
    """
    # Initialize the translation list
    translations = []
    if model_type.lower() == "t5":
        # For T5, use the pipeline task pattern
        task = f"translation_{source_lang_code}_to_{target_lang_code}"
        translator = pipeline(task, model=model_name, device=device)
        for text in tqdm(source_texts, desc="Translating"):
            translation = translator(text)
            print(translation[0]['translation_text'])
            translations.append(translation[0]['translation_text'])
    elif model_type.lower() in ["mbart", "marian", "m2m100", "nllb"]:
        # For mBART/MarianMT/nllb/m2m100, use src_lang and tgt_lang
        translator = pipeline("translation", model=model_name, device=device)
        for text in tqdm(source_texts, desc="Translating"):
            translation = translator(text, src_lang=source_lang_code, tgt_lang=target_lang_code)
            print(translation[0]['translation_text'])
            translations.append(translation[0]['translation_text'])
    else:
        # Default: try pipeline with just the model
        translator = pipeline("translation", model=model_name, device=device)
        for text in tqdm(source_texts, desc="Translating"):
            translation = translator(text)
            print(translation[0]['translation_text'])
            translations.append(translation[0]['translation_text'])

    return translations


def save_model_results(model_name, source_texts, translations, references, bleu_score, output_file):
    """
    Save results for a model in a single output file, with sections for each model.
    """
    col_width = 50
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== {model_name} ===\n")
        header = f"{'source':<{col_width}}{'translation':<{col_width}}{'reference':<{col_width}}\n"
        f.write(header)
        for src, trans, ref in zip(source_texts, translations, references):
            f.write(f"{src:<{col_width}}{trans:<{col_width}}{ref:<{col_width}}\n")
        f.write(f"BLEU score: {bleu_score}\n")

def save_bleu_score(tsv_file, source_column, target_column, models, line_counts, bleu_file, translations_file, device=0):
    max_line_count = max(line_counts)
    # Clear files before writing
    open(bleu_file, "w").close()
    open(translations_file, "w").close()
    # Set column widths
    col_widths = [8, 45, 12]  # lines, model, bleu_score
    header_fmt = f"{{:<{col_widths[0]}}}{{:<{col_widths[1]}}}{{:<{col_widths[2]}}}\n"
    row_fmt = f"{{:<{col_widths[0]}}}{{:<{col_widths[1]}}}{{:<{col_widths[2]}.2f}}\n"
    with open(bleu_file, "w", encoding="utf-8") as bf:
        bf.write(header_fmt.format("lines", "model", "bleu_score"))
    #for num_lines in line_counts:
    source_texts, references = extract_source_reference(tsv_file, source_column, target_column, num_sentences=max_line_count)
    for model_name, model_type, src_code, tgt_code in models:
        print(f"Translating {max_line_count} lines with {model_name}...")
        translations = translate_sentences(
            source_texts,
            model_name,
            model_type=model_type,
            source_lang_code=src_code,
            target_lang_code=tgt_code,
            device=device
        )
        for line_count in line_counts: 
            bleu = sacrebleu.corpus_bleu(translations[:line_count], [references[:line_count]])
            print(f"BLEU score for {model_name} ({line_count} lines): {bleu.score}")
            # Save BLEU score
            with open(bleu_file, "a", encoding="utf-8") as bf:
                bf.write(row_fmt.format(line_count, model_name, bleu.score))
            # Save translations only for the max line count
            if line_count == max_line_count:
                save_model_results(model_name, source_texts, translations, references, bleu.score, translations_file)


def main():
    tsv_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Data/train.clean.en-de.tsv"
    source_column = "en"
    target_column = "de"
    translations_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Model_Exploration/translation_pipeline_output.tsv"
    bleu_file = "/home/mlt_ml2/ML_Applied_Project_2025S/Model_Exploration/model_comparison.tsv"
    models = [
        ("google-t5/t5-base", "t5", "en", "de"),
        ("Helsinki-NLP/opus-mt-en-de", "marian", "en", "de"),
        ("facebook/nllb-200-distilled-600M", "nllb", "eng_Latn", "deu_Latn"),
        ("facebook/mbart-large-50-many-to-many-mmt", "mbart", "en_XX", "de_DE"),
        ("DunnBC22/mbart-large-50-English_German_Translation", "mbart", "en_XX", "de_DE"),
        ("Tanhim/translation-En2De","marian", "en", "de"),
        ("facebook/m2m100_418M", "m2m100", "en", "de")
    ]
    # Define the number of lines to test
    line_counts = [10, 100, 200]
    save_bleu_score(tsv_file, source_column, target_column, models, line_counts, bleu_file, translations_file, device=0)
    
if __name__ == "__main__":
    main()