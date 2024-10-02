from transformers import pipeline

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze text
text = "I love using Hugging Face models!"
result = sentiment_analyzer(text)

print(result)