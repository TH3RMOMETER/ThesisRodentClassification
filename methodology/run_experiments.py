from config import Config
import pandas as pd
from torchaudio_data import (
    create_df_from_audio_filepaths,
    experiments,
    fix_gpu,
    train_keras_network,
    train_torch_network,
)

if __name__ == "__main__":
    config = Config()
    fix_gpu()
    # create experiment config list
    experiment_config_list = experiments()
    # create audio dataframe from audio filepaths
    audio_df = create_df_from_audio_filepaths(config)
    # load in train and test audio dataframes
    train_audio_df = pd.read_csv("train_audio.csv")
    test_audio_df = pd.read_csv("test_audio.csv")
    # run experiments
    for experiment_config in experiment_config_list:
        if experiment_config.model_type == "yolo" or experiment_config.model_type == "long_yolo":
            train_keras_network(experiment_config, train_audio_df, test_audio_df)
        elif experiment_config.model_type == "resnet":
            train_torch_network(experiment_config, train_audio_df, test_audio_df)
