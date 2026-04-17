import torch

class BaseConfig:
    num_classes = 7
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    weights = None
    name = "generic"

class RAFDBConfig(BaseConfig):
    name = "rafdb"
    folder_to_idx = {'6':0, '3':1, '2':2, '4':3, '7':4, '5':5, '1':6}
    weights = torch.tensor([1.5, 4.0, 5.0, 0.4, 1.0, 1.2, 1.8])

class FER2013Config(BaseConfig):
    name = "fer2013"
    # Matches your folder names in image_64e686.png
    folder_to_idx = {
        'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 
        'neutral': 4, 'sad': 5, 'surprise': 6
    }
    weights = None

def get_dataset_config(name):
    name = name.lower()
    if "raf" in name: return RAFDBConfig
    if "fer2013" in name: return FER2013Config
    return BaseConfig