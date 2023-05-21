from dataclasses import dataclass
import multiprocessing
import torch
from torchvision.models import resnet50, ResNet50_Weights
import yoloNetwork
import LongyoloNetwork
from ast_models import ASTModel


@dataclass
class Config(object):
    """Configuration class"""
    model_type: str = "resnet"
    delta: bool = False
    delta_delta: bool = True
    num_classes: int = 1
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    max_epochs: int = 100
    max_lr: float = 0.01
    audio_length: int = 120
    audio_data_folder_path: str = (
        r"C:\Users\gijst\Documents\Master Data Science\Thesis\audio_data"
    )
    agouti_media_path: str = r"C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\media.csv"
    agouti_observations_path: str = r"C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\observations.csv"
    slice_len: int = 25
    step_size: float = 0.9
    cropped_audio_path = r"C:\Users\gijst\Documents\Master Data Science\Thesis\audio_data\cropped_audio_data"
    output_path: str = (
        r"C:\Users\gijst\Documents\Master Data Science\Thesis\processed_data"
    )
    # get number of cpus
    num_cpus: int = multiprocessing.cpu_count()


    def create_network(self):
        """Create network"""
        if self.model_type == "yolo":
            return yoloNetwork.load_model(load_weights=True)
        elif self.model_type == "long_yolo":
            return LongyoloNetwork.load_model(load_weights=True)
        elif self.model_type == "resnet":
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif self.model_type == "ast_model":
            # input shape = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
            return ASTModel(label_dim=2, input_fdim=52, input_tdim=37648, audioset_pretrain=True)
