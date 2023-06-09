from dataclasses import dataclass


@dataclass
class Config(object):
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    max_epochs: int = 100
    max_lr: float = 0.01
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

""" 
def run(config: Config):
    df = create_df_from_audio_filepaths(config)
    dataset = AudioDataset(df=df, audio_length=5)
    model = AudioModel()
    x = 1


# if main run
if __name__ == "__main__":
    run(Config())
 """