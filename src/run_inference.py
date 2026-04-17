import os
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from model import FERFullPipeline

def run_team_inference():
    # 1. Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/content/drive/MyDrive/FER_Checkpoints/best_emotion_model.pt"
    image_dir = "test_images"
    
    # CRITICAL: This MUST be alphabetical order to match training
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # 2. Initialize and Load Weights
    model = FERFullPipeline(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Iterate and Predict
    print(f"{'IMAGE':<20} | {'PREDICTION':<12} | {'CONFIDENCE'}")
    print("-" * 50)
    
    for img_file in sorted(os.listdir(image_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            
            # Process
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            print(f"{img_file:<20} | {emotion_labels[pred_idx.item()]:<12} | {conf.item()*100:.2f}%")

# Execute
run_team_inference()