import os
import torch
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def align_and_crop_dataset(raw_dir: Path, aligned_dir: Path, image_size: int = 224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing MTCNN on: {device}")
    
    mtcnn = MTCNN(
        image_size=image_size, 
        margin=20, 
        keep_all=False, 
        post_process=False, 
        device=device
    )

    aligned_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through class folders (e.g., Happy, Sad)
    for class_folder in raw_dir.iterdir():
        if not class_folder.is_dir():
            continue
            
        output_class_folder = aligned_dir / class_folder.name
        output_class_folder.mkdir(parents=True, exist_ok=True)

        image_paths = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        
        for img_path in tqdm(image_paths, desc=f"Aligning {class_folder.name}"):
            try:
                img = Image.open(img_path).convert('RGB')
                save_path = str(output_class_folder / img_path.name)
                
                # Detect and save the cropped face
                mtcnn(img, save_path=save_path)
                
            except Exception as e:
                pass # Skip images where a face cannot be detected

if __name__ == "__main__":
    # Update these paths to match your Google Drive setup
    RAW_DATASET_PATH = Path("/content/drive/MyDrive/Datasets/Balanced_FER_Raw")
    ALIGNED_DATASET_PATH = Path("/content/drive/MyDrive/Datasets/Balanced_FER_Aligned")
    
    align_and_crop_dataset(RAW_DATASET_PATH, ALIGNED_DATASET_PATH)