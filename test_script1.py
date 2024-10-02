from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input
text = "I'm excited to learn about Hugging Face!"
inputs = tokenizer(text, return_tensors="pt")

# Get prediction
import torch
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = torch.argmax(logits, dim=1).item()
print(f"Predicted class: {predicted_class}")