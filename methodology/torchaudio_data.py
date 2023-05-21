import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Audio processing
import torchaudio
import torchaudio.transforms as T
from config import Config
from lightning.pytorch.loggers import WandbLogger
from module import DataGenerator
from tensorflow import keras
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from tqdm.notebook import trange
from wandb.keras import WandbCallback

# Pre-trained image models
import wandb


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
        spectogram=False,
        long=False,
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
        self.spectogram = spectogram
        self.long = long

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

        if self.spectogram:
            # nftt in seconds
            nftt = 0.0032
            overlap = 0.0028
            window = 0.0032
            if self.long:
                nftt = 0.01
                overlap = 0.005
                window = 0.01
            # convert parameters in seconds to frames
            nftt = int(nftt * self.target_sample_rate)
            overlap = int(overlap * self.target_sample_rate)
            window = int(window * self.target_sample_rate)
            # convert audio to spectogram
            spec = torchaudio.transforms.Spectrogram(
                n_fft=nftt, win_length=window, hop_length=window
            )(audio)
            return (torch.stack([spec]), torch.tensor(self.labels[index]).float())

        # Convert to Mel spectrogram
        melspectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_mels=self.n_mfcc
        )
        melspec = melspectrogram(audio)

        if self.delta:
            # Add delta
            melspec = torchaudio.transforms.ComputeDeltas()(melspec)
        elif self.delta_delta:
            # Add delta delta
            melspec = torchaudio.transforms.ComputeDeltas()(melspec)
            melspec = torchaudio.transforms.ComputeDeltas()(melspec)

        # Add any data augmentations for spectrogram you like here
        # (e.g., Mixup, cutmix, time masking, frequency masking)
        ...

        return (torch.stack([melspec]), torch.tensor(self.labels[index]).float())


class AudioModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        input_shape=(1, 40, 7501),
        config: Config = Config(),
    ):
        super().__init__()

        # self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        self.model = config.create_network()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, num_classes))  # type: ignore
        # self.model[-1] = nn.Sequential(nn.Linear(256, num_classes))  # type: ignore
        """ self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(self.in_features, num_classes)) """
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
        # set last n layers to be trainable

    def forward(self, images):
        # return model output trough sigmoid
        logits = self.model(images)  # type: ignore
        logits = torch.sigmoid(logits)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss, acc = self._get_preds_loss_accuracy(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y.unsqueeze(1))
        acc = accuracy(logits, y.unsqueeze(1), task="binary")
        return logits, loss, acc
    


def create_model(config: Config, audio_shape: tuple):
    if config.model_type == "yolo":
        yolo_network = config.create_network()
        yolo_network.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=["accuracy"],
        )  # type: ignore
        return yolo_network
    elif config.model_type == "resnet":
        return AudioModel(input_shape=audio_shape)
    return


def create_train_test_val_split(audio_df: AudioDataset) -> tuple:
    train_size = int(0.8 * len(audio_df))
    val_size = int(0.1 * len(audio_df))
    test_size = len(audio_df) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(  # type: ignore
        audio_df, [train_size, val_size, test_size]
    )
    return train_set, val_set, test_set


def train_keras_network(config: Config):
    audio_df = create_df_from_audio_filepaths(config)
    audio_df = AudioDataset(audio_df, config.audio_length, spectogram=True)
    # split dataset
    train_set, val_set, test_set = create_train_test_val_split(audio_df)
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False)

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = audio.shape

    # create wandb logger
    run = wandb.init(project="test_rat_USV", reinit=True)

    model = create_model(config, audio_shape)
    # convert torch loader to keras compatible object
    train_loader = DataGenerator(train_loader, 2)
    val_loader = DataGenerator(val_loader, 2)
    model.fit_generator(train_loader, epochs=10, validation_data=val_loader, callbacks=[WandbCallback()])  # type: ignore
    run.finish()


def train_torch_network(config: Config):
    num_cpus = config.num_cpus
    # create dataset
    audio_df = create_df_from_audio_filepaths(config)
    audio_df = AudioDataset(
        audio_df,
        config.audio_length,
        delta=config.delta,
        delta_delta=config.delta_delta,
    )
    # split dataset
    train_set, val_set, test_set = create_train_test_val_split(audio_df)

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = audio.shape

    # create wandb logger
    wandb_logger = WandbLogger(project="test_rat_USV", log_model="all")

    # create model and trainer
    audio_model = AudioModel(input_shape=audio_shape)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu",
    )

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # train the model
    trainer.fit(model=audio_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # stop wandb run


if __name__ == "__main__":
    config = Config()
    if config.model_type == "yolo" or config.model_type == "long_yolo":
        train_keras_network(config)
    elif config.model_type == "resnet":
        train_torch_network(config)
