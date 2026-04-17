import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from src.model import FERFullPipeline

def predict_emotion(image_path, model_path):
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Define standard labels (Update based on your dataset order)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # 3. Initialize model and load weights
    model = FERFullPipeline(num_classes=7).to(device)
    # weights_only=True is a security best practice for loading .pt files
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Switch to evaluation mode
    
    # 4. Image Preprocessing (Matches your Kaggle Dataloader logic)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 5. Load and process image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
    
    # 6. Run Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    # 7. Output Result
    emotion = emotion_labels[predicted_idx.item()]
    print(f"--- Result ---")
    print(f"Prediction: {emotion}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")

if __name__ == "__main__":
    # Test on a single image
    TEST_IMAGE = "path/to/your/test_image.jpg"
    MODEL_WEIGHTS = "best.pt"
    
    if Path(TEST_IMAGE).exists() and Path(MODEL_WEIGHTS).exists():
        predict_emotion(TEST_IMAGE, MODEL_WEIGHTS)
    else:
        print("Error: Ensure image and model weights paths are correct.")