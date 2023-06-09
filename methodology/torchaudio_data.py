import os
import sys

import librosa
import matplotlib.pyplot as plt
import torchmetrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from lightning.pytorch.callbacks import ModelCheckpoint as pl_ModelCheckpoint
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, normalize
from transformers import ASTForAudioClassification, AutoFeatureExtractor
from wandb.keras import WandbCallback

sys.path.append(r"G:\thesis\ThesisRodentClassification")

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Audio processing
import torchaudio
import torchaudio.transforms as T
from config import Config

# from ast_master.src.dataloader import AudiosetDataset
# from ast_master.src.models import ASTModel
from lightning.pytorch.loggers import WandbLogger
from module import DataGenerator, f1_m, precision_m, recall_m
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

import wandb


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def create_df_from_audio_filepaths(config: Config):
    """Create dataframe with audio filepaths and labels"""
    folder_path = config.cropped_audio_path
    # if sub folder is noise set to 0, if folder is rats set to 1
    labels = []
    filepaths = []
    # go trough all subfolders
    for subfolder in os.listdir(folder_path):
        # if subfolder is noise set to 0
        if subfolder == "noise":
            for file in os.listdir(os.path.join(folder_path, subfolder)):
                labels.append(0)
                filepaths.append(os.path.join(folder_path, subfolder, file))
        # if subfolder is rats set to 1
        elif subfolder == "rats":
            for file in os.listdir(os.path.join(folder_path, subfolder)):
                labels.append(1)
                filepaths.append(os.path.join(folder_path, subfolder, file))

    # create dataframe
    df = pd.DataFrame({"file_path": filepaths, "label": labels})
    return df


class AudioDataset(Dataset):
    def __init__(
        self,
        df,
        audio_length,
        target_sample_rate=32000,
        n_mfcc=40,
        wave_transforms=None,
        spec_transforms=None,
        delta=False,
        delta_delta=False,
        spectrogram=False,
        long=False,
        model_type="test",
    ):
        self.df = df
        self.file_paths = df["file_path"].values
        self.labels = df["label"].values
        self.audio_length = audio_length
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate * audio_length
        self.n_mfcc = n_mfcc
        self.wave_transforms = wave_transforms
        self.spec_transforms = spec_transforms
        self.delta = delta
        self.delta_delta = delta_delta
        self.spectrogram = spectrogram
        self.long = long
        self.model_type = model_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load audio from file to waveform
        audio, sample_rate = torchaudio.load(self.file_paths[index])  # type: ignore

        # Convert to mono
        audio = torch.mean(audio, axis=0)  # type: ignore

        # Resample
        if sample_rate != self.target_sample_rate:
            resample = T.Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)

        # Adjust number of samples
        if audio.shape[0] > self.num_samples:
            # Crop
            audio = audio[: self.num_samples]
        elif audio.shape[0] < self.num_samples:
            # Pad
            audio = F.pad(audio, (0, self.num_samples - audio.shape[0]))

        if self.model_type == "ast":
            return audio, torch.tensor(self.labels[index]).float()

        if self.spectrogram:
            # nftt in seconds
            nftt = 0.0032
            overlap = 0.0028
            window = 0.0032
            low_cutoff = 0.3
            if self.long:
                nftt = 0.01
                overlap = 0.005
                window = 0.01
            # convert parameters in seconds to frames
            nftt = int(
                nftt * self.target_sample_rate
            )  # 4 for  minutes of audio file, 2 for compensating for slice length
            overlap = int(overlap * self.target_sample_rate)
            window = int(window * self.target_sample_rate)
            # convert audio to spectrogram
            spec = torchaudio.transforms.Spectrogram(n_fft=nftt, normalized=True)(audio)

            return torch.stack([spec]), torch.tensor(self.labels[index]).float()

        # Convert to MFCC
        nftt = 0.0032
        nmfcc = int(
            nftt * self.target_sample_rate
        )  # 4 for  minutes of audio file, 2 for compensating for slice length
        mfcc = T.MFCC(
            sample_rate=self.target_sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_mels": 64},
        )
        mfcc = mfcc(audio)

        if self.delta:
            # Add delta
            mfcc = torchaudio.transforms.ComputeDeltas()(mfcc)
        elif self.delta_delta:
            # Add delta delta
            mfcc = torchaudio.transforms.ComputeDeltas()(mfcc)
            mfcc = torchaudio.transforms.ComputeDeltas()(mfcc)

        return torch.stack([mfcc]), torch.tensor(self.labels[index]).float()


class AudioModel(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        num_classes: int,
        input_shape
    ):
        super().__init__()
        self.model = config.create_network()
        in_features = self.model.fc.in_features  # type: ignore
        self.model.fc = nn.Sequential(nn.Linear(in_features, num_classes))  # type: ignore
        first_conv_layer = nn.Conv2d(
            input_shape[0],
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.model = nn.Sequential(first_conv_layer, self.model)  # type: ignore

        self.metric_collection = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "f1": BinaryF1Score(),
                "auc_roc": BinaryAUROC(),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
            }
        )
        self.metric_collection_val = torchmetrics.MetricCollection(
            {
                "val_accuracy": torchmetrics.Accuracy(task="binary"),
                "val_f1": BinaryF1Score(),
                "auc_roc": BinaryAUROC(),
                "val_precision": torchmetrics.Precision(task="binary"),
                "val_recall": torchmetrics.Recall(task="binary"),
            }
        )

        self.save_hyperparameters()

    def forward(self, images):
        # return model output trough sigmoid
        logits = self.model(images)  # type: ignore
        logits = torch.sigmoid(logits)
        return logits

    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.binary_cross_entropy(logits, ys)
        return logits, loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        reduce_lr_on_plateau = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": reduce_lr_on_plateau,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.metric_collection.update(logits, y.unsqueeze(1))
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log_dict(self.metric_collection, on_step=True, on_epoch=False)  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.metric_collection_val.update(logits, y.unsqueeze(1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.metric_collection_val, on_step=False, on_epoch=True, prog_bar=True  # type: ignore
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.metric_collection.update(logits, y.unsqueeze(1))
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.metric_collection, on_step=False, on_epoch=True, prog_bar=True  # type: ignore
        )


def create_model(config: Config, audio_shape: list):
    if config.model_type == "yolo" or "long_yolo":
        yolo_network = config.create_network(shape=audio_shape)
        yolo_network.compile(  # type: ignore
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[
                keras.metrics.BinaryAccuracy(),
                keras.metrics.AUC(),
                f1_m,
                precision_m,
                recall_m,
            ],
        )  # type: ignore
        return yolo_network
    elif config.model_type == "resnet":
        return AudioModel(input_shape=audio_shape, config=config, num_classes=1)
    elif config.model_type == "ast_model":
        ast_network = config.create_network()
        return ast_network
    return


def create_train_val_split(audio_df: AudioDataset, random_state: int = 42) -> tuple:
    torch.manual_seed(random_state)
    train_size = int(0.8 * len(audio_df))
    val_size = len(audio_df) - train_size
    train_set, val_set = torch.utils.data.random_split(  # type: ignore
        audio_df, [train_size, val_size]
    )
    return train_set, val_set


def create_ast_data_and_model(config: Config):
    audio_conf = {
        "num_mel_bins": 64,
        "target_length": config.audio_length,
        "freqm": 0,
        "timem": 0,
        "mixup": 0,
        "mode": "train",
        "mean": 4.2677393,
        "std": 4.5689974,
        "noise": False,
    }
    audio_df = create_df_from_audio_filepaths(config)
    audio_data_set = AudiosetDataset(
        audio_df, audio_conf, target_sample_rate=config.target_sample_rate
    )
    train_set, val_set, test_set = create_train_val_split(audio_data_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # convert audio_length from seconds to frames using sample rate
    audio_length = config.audio_length * config.target_sample_rate

    audio_model = ASTModel(
        label_dim=1,
        input_fdim=128,
        input_tdim=config.audio_length,
        imagenet_pretrain=True,
        audioset_pretrain=True,
    )

    return audio_model, train_loader, val_loader, test_loader


def train_ast_model(config: Config, train_audio_df: pd.DataFrame):
    # audio_model, train_loader, val_loader, test_loader = create_ast_data_and_model(
    #  config
    # )

    audio_model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    audio_df = AudioDataset(
        train_audio_df,
        config.audio_length,
        spectrogram=config.spectrogram,
        delta=config.delta,
        delta_delta=config.delta_delta,
    )
    # split dataset
    train_set, val_set = create_train_val_split(audio_df)
    # create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # wandb_logger = WandbLogger(project="test_rat_USV", reinit=True)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        # logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu",
    )
    trainer.fit(audio_model, train_loader, val_loader)
    trainer.test(audio_model, test_loader)


def train_keras_network(
    config: Config, train_audio_df: pd.DataFrame, test_audio_df: pd.DataFrame
):
    """function for training YOLO and long YOLO pretrained models

    Args:
        config (Config): experiment setyp
        train_audio_df (pd.DataFrame): training dataframe
    """
    audio_df = AudioDataset(
        train_audio_df,
        config.audio_length,
        spectrogram=config.spectrogram,
        delta=config.delta,
        delta_delta=config.delta_delta,
        n_mfcc=config.n_mfcc,
        target_sample_rate=config.target_sample_rate,
    )
    test_audio_set = AudioDataset(
        test_audio_df,
        config.audio_length,
        spectrogram=config.spectrogram,
        delta=config.delta,
        delta_delta=config.delta_delta,
        n_mfcc=config.n_mfcc,
        target_sample_rate=config.target_sample_rate,
    )

    # split dataset
    train_set, val_set = create_train_val_split(audio_df)
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_audio_set, batch_size=config.batch_size, shuffle=False
    )

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = list(audio.shape)
    audio_shape.reverse()
    # swap first two dimensions
    audio_shape[0], audio_shape[1] = audio_shape[1], audio_shape[0]

    # create wandb logger
    run = wandb.init(
        project="test_rat_USV",
        reinit=True,
        name=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
        config=config.__dict__,
    )
    callbacks = [
        WandbCallback(save_model=False),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
        ModelCheckpoint(
            filepath=f"models/{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}.hdf5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        ),
    ]

    model = create_model(config, audio_shape)
    # convert torch loader to keras compatible object
    train_loader = DataGenerator(train_loader, 2)
    val_loader = DataGenerator(val_loader, 2)
    test_loader = DataGenerator(test_loader, 2)
    model.fit(train_loader, epochs=config.epochs, validation_data=val_loader, callbacks=callbacks)  # type: ignore
    test_stats = model.evaluate(test_loader, callbacks=WandbCallback())  # type: ignore
    # get model metrics
    metrics = {}
    for i, metric in enumerate(model.metrics_names):
        metrics[f'test_{metric}'] = test_stats[i]
    wandb.log(metrics)
    
    run.finish()  # type: ignore


def train_torch_network(
    config: Config, train_audio_df: pd.DataFrame, test_audio_df: pd.DataFrame
):
    num_cpus = config.num_cpus
    torch.set_float32_matmul_precision("high")
    # create dataset
    audio_df = AudioDataset(
        train_audio_df,
        config.audio_length,
        delta=config.delta,
        delta_delta=config.delta_delta,
        target_sample_rate=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
    )
    test_audio_set = AudioDataset(
        test_audio_df,
        config.audio_length,
        delta=config.delta,
        delta_delta=config.delta_delta,
        target_sample_rate=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
    )
    # split dataset
    train_set, val_set = create_train_val_split(audio_df)

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = audio.shape

    # create wandb logger
    wandb_logger = WandbLogger(
        project="test_rat_USV",
        log_model=True,
        name=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
    )

    # create modelcheckpoint callback
    checkpoint_callback = pl_ModelCheckpoint(
        dirpath=f"models/",
        filename=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
        monitor="val_loss",
        verbose=True,
    )
    # create model and trainer
    audio_model = AudioModel(input_shape=audio_shape, config=config, num_classes=1)

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_audio_set, batch_size=config.batch_size, shuffle=False
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
    )
    # train the model
    trainer.fit(
        model=audio_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    test = trainer.test(model=audio_model, dataloaders=test_loader)
    # add test in front of metrics if test is not in front
    test[0] = {f"test_{key}": value for key, value in test[0].items() if "test" not in key}
    wandb_logger.log_metrics(test[0])
    wandb_logger.experiment.finish()


def experiments():
    yolo_spec = Config(model_type="yolo", spectrogram=True)
    yolo_mfcc = Config(model_type="yolo", spectrogram=False)
    yolo_mfcc_delta = Config(model_type="yolo", spectrogram=False, delta=True)
    yolo_mfcc_delta_delta = Config(
        model_type="yolo", spectrogram=False, delta=False, delta_delta=True
    )
    long_yolo_spec = Config(model_type="long_yolo", spectrogram=True)
    long_yolo_mfcc = Config(model_type="long_yolo", spectrogram=False)
    long_yolo_mfcc_delta = Config(model_type="long_yolo", spectrogram=False, delta=True)
    long_yolo_mfcc_delta_delta = Config(
        model_type="long_yolo", spectrogram=False, delta=False, delta_delta=True
    )
    resnet_spec = Config(model_type="resnet", spectrogram=True)
    resnet_mfcc = Config(model_type="resnet", spectrogram=False)
    resnet_mfcc_delta = Config(model_type="resnet", spectrogram=False, delta=True)
    resnet_mfcc_delta_delta = Config(
        model_type="resnet", spectrogram=False, delta=False, delta_delta=True
    )

    """  experiment_list = [
        yolo_spec,
        yolo_mfcc,
        yolo_mfcc_delta,
        yolo_mfcc_delta_delta,
        long_yolo_spec,
        long_yolo_mfcc,
        long_yolo_mfcc_delta,
        long_yolo_mfcc_delta_delta,
        resnet_spec,
        resnet_mfcc,
        resnet_mfcc_delta,
        resnet_mfcc_delta_delta,
    ] """

    experiment_list = [
        resnet_spec,
        resnet_mfcc,
        resnet_mfcc_delta,
        resnet_mfcc_delta_delta
        ]
    return experiment_list


if __name__ == "__main__":
    config = Config()
    fix_gpu()
