import torch

class BaseConfig:
    """Default settings for our internal Alphabetical Standard."""
    num_classes = 7
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    weights = None
    folder_to_idx = None # Default assumes ImageFolder alphabetical sorting

class RAFDBConfig(BaseConfig):
    """Config for the imbalanced and numbered RAF-DB dataset."""
    name = "rafdb"
    # Maps RAF-DB folder numbers to our internal indices
    # 1:Surprise, 2:Fear, 3:Disgust, 4:Happy, 5:Sad, 6:Angry, 7:Neutral
    folder_to_idx = {
        '6': 0, # Angry
        '3': 1, # Disgust
        '2': 2, # Fear
        '4': 3, # Happy
        '7': 4, # Neutral
        '5': 5, # Sad
        '1': 6  # Surprise
    }
    # Weighted loss to stop the model from only guessing 'Happy'
    weights = torch.tensor([1.5, 4.0, 5.0, 0.4, 1.0, 1.2, 1.8])

class FER2013Config(BaseConfig):
    """Config for the Grayscale FER2013 dataset."""
    name = "fer2013"
    # Since your folders are named 'angry', 'disgust', etc., 
    # PyTorch sorts them alphabetically which matches our labels!
    folder_to_idx = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    # FER2013 is still slightly imbalanced (Disgust is tiny),
    # but we can start without weights to see the 'Generalization' first.
    weights = None 

def get_dataset_config(name):
    """Factory to pick the right DNA for the training run."""
    name = name.lower()
    if "raf" in name:
        return RAFDBConfig
    elif "fer2013" in name:
        return FER2013Config
    return BaseConfig