import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_text_from_frame(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, return_tensors="pt")
    
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption

def process_video(video_path, capture_interval=5):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * capture_interval)
    
    frame_count = 0
    last_caption = ""
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            caption = extract_text_from_frame(frame)
            
            if caption != last_caption:
                print(f"Frame {frame_count} (Time: {frame_count/fps:.2f}s): {caption}")
                last_caption = caption

    video.release()

# Usage
video_path = 'sample_video.mp4'
process_video(video_path, capture_interval=2, confidence_threshold=0.6)