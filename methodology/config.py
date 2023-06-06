import multiprocessing
from dataclasses import dataclass

from torchvision.models import resnet50, ResNet50_Weights

import LongyoloNetwork
import yoloNetwork
from ast_models import ASTModel


@dataclass
class Config(object):
    """Configuration class"""
    model_type: str = "ast_model"
    spectrogram: bool = True
    delta: bool = True
    delta_delta: bool = False
    num_classes: int = 1
    batch_size: int = 1
    epochs: int = 10
    learning_rate: float = 0.001
    max_epochs: int = 100
    max_lr: float = 0.01
    audio_length: int = 240
    audio_data_folder_path: str = (
        r"C:\Users\gijst\Documents\Master Data Science\Thesis\audio_data"
    )
    agouti_media_path: str = r"C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\media.csv"
    agouti_observations_path: str = r"C:\Users\gijst\Documents\Master Data Science\Thesis\flevopark-20230202124032\observations.csv"
    agouti_filepath = r"C:\Users\gijst\Documents\Master Data Science\Thesis\agouti_data\agouti.pkl"
    audio_folder = r"C:\Users\gijst\Documents\Master Data Science\Thesis\audio_data\data\20230417"
    slice_len: int = 25
    step_size: float = 0.9
    cropped_audio_path = r"G:\thesis\ThesisRodentClassification\processed_data"
    output_path: str = (
        r"C:\Users\gijst\Documents\Master Data Science\Thesis\processed_data"
    )
    # get number of cpus
    num_cpus: int = multiprocessing.cpu_count()
    target_sample_rate: int = 32000

    def create_network(self, shape=None):
        """Create network"""
        if shape is None:
            shape = [37648, 52, 1]
        if self.model_type == "yolo":
            return yoloNetwork.load_model(shape=shape, load_weights=True)
        elif self.model_type == "long_yolo":
            return LongyoloNetwork.load_model(shape=shape, load_weights=True)
        elif self.model_type == "resnet":
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif self.model_type == "ast_model":
            # input shape = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
            return ASTModel(label_dim=1, input_fdim=52, input_tdim=37648, audioset_pretrain=True)
