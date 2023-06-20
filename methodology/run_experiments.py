from config import Config
from sklearn.model_selection import train_test_split
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
    # split dataframe into train and test
    train_audio_df, test_audio_df = train_test_split(
        audio_df, test_size=0.1, random_state=config.random_state
    )
    # run experiments
    for experiment_config in experiment_config_list:
        if experiment_config.model_type == "yolo" or experiment_config.model_type == "long_yolo":
            train_keras_network(experiment_config, train_audio_df, test_audio_df)
        elif experiment_config.model_type == "resnet":
            train_torch_network(experiment_config, train_audio_df, test_audio_df)
