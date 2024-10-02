import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def load_image(image_path):
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def read_text_from_image(image, model, processor):
    prompt = "Read and transcribe all the text in this image:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return generated_text.replace(prompt, "").strip()

if __name__ == "__main__":
    print("Loading Florence-2 model and processor...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/florence-2-large", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("microsoft/florence-2-large")
    print("Model and processor loaded successfully.")

    image_path = input("Enter the path of the image: ")
    image = load_image(image_path)

    if image:
        print("Processing image...")
        extracted_text = read_text_from_image(image, model, processor)
        print("\nExtracted Text:")
        print(extracted_text)
    else:
        print("Failed to load the image.")

print("Script execution completed.")