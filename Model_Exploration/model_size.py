from transformers import AutoModelForSeq2SeqLM

def print_model_size(model_name: str):
    """
    Loads a Hugging Face Seq2Seq model and prints its total and trainable parameter counts.
    
    Args:
        model_name (str): The model identifier on Hugging Face hub.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

# Example usage:
models = [
    "google-t5/t5-base",
    "Helsinki-NLP/opus-mt-en-de",
    "facebook/nllb-200-distilled-600M",
    "facebook/mbart-large-50-many-to-many-mmt",
    "DunnBC22/mbart-large-50-English_German_Translation",
    "Tanhim/translation-En2De",
    "facebook/m2m100_418M"
]

for model_name in models:
    print_model_size(model_name)
