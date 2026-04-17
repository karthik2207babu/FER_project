import torch

class BaseConfig:
    num_classes = 7
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    weights = None

class RAFDBConfig(BaseConfig):
    name = "rafdb"
    # MAPS FOLDER NUMBERS TO OUR MODEL'S INDICES:
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
    # Weighting: Give more importance to rare classes (Disgust, Fear)
    weights = torch.tensor([1.5, 4.0, 5.0, 0.4, 1.0, 1.2, 1.8])

def get_dataset_config(name):
    if name.lower() == "rafdb":
        return RAFDBConfig
    return BaseConfig