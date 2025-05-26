from transformers import pipeline
#import sacrebleu

# Use a pipeline as a high-level helper
from transformers import pipeline

translator = pipeline("text2text-generation", model="google/mt5-base")
text = "translate English to German: The house is wonderful."
result = translator(text)
print(result[0]['generated_text'])