import torch

class BaseConfig:
    """Default settings for any dataset."""
    num_classes = 7
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    weights = None  # Default is equal weights

class BalancedConfig(BaseConfig):
    """Config for your 82.95% accuracy balanced dataset."""
    name = "balanced_general"
    # No weights needed because the dataset is already balanced
    weights = None 

class RAFDBConfig(BaseConfig):
    """Config for the imbalanced RAF-DB dataset."""
    name = "raf_db"
    # Calculated weights to help the model learn rare emotions (Disgust/Fear)
    weights = torch.tensor([1.5, 4.0, 5.0, 0.4, 1.0, 1.2, 1.8])

# Factory to get the right config
def get_dataset_config(name):
    configs = {
        "balanced": BalancedConfig,
        "rafdb": RAFDBConfig
    }
    return configs.get(name.lower(), BaseConfig)