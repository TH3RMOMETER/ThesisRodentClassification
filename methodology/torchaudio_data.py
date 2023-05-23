from doctest import master
from email.mime import audio
import os

import sys
sys.path.append(r"C:\Users\gijst\Documents\Master Data Science\Thesis")

import lightning.pytorch as pl
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
from ast_master.src.dataloader import AudiosetDataset
from ast_master.src.models import ASTModel
from lightning.pytorch.loggers import WandbLogger
from module import DataGenerator, ImagePredictionLogger, f1_m, precision_m, recall_m
from tensorflow import keras
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, precision_recall
from torchmetrics.functional import accuracy
from wandb.keras import WandbCallback

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

        return (torch.stack([melspec]), torch.tensor(self.labels[index]).float())


class AudioModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        input_shape=(1, 40, 7501),
        config: Config = Config(),
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

        # metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()
        self.val_auc_roc = BinaryAUROC()
        self.test_auc_roc = BinaryAUROC()

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
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.train_acc(logits, y.unsqueeze(1))
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.val_acc(logits, y.unsqueeze(1))
        precision = precision_recall.BinaryPrecision()(logits, y.unsqueeze(1))
        recall = precision_recall.BinaryRecall()(logits, y.unsqueeze(1))
        f1_score = BinaryF1Score()(logits, batch[1].unsqueeze(1))
        auc_roc = BinaryAUROC()(logits, batch[1].unsqueeze(1))
        self.log("val_auc_roc", auc_roc, on_step=False, on_epoch=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)
        self.log("val_f1", f1_score, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        return logits

    def on_validation_epoch_end(self, validation_step_outputs):
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(  # type: ignore
            {
                "valid/logits": wandb.Histogram(flattened_logits.to("cpu")),  # type: ignore
                "global_step": self.global_step,
            }
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        self.test_acc(logits, y.unsqueeze(1))
        precision = precision_recall.BinaryPrecision()(logits, y.unsqueeze(1))
        recall = precision_recall.BinaryRecall()(logits, y.unsqueeze(1))
        f1_score = BinaryF1Score()(logits, batch[1].unsqueeze(1))
        auc_roc = BinaryAUROC()(logits, batch[1].unsqueeze(1))
        self.log("test_auc_roc", auc_roc, on_step=False, on_epoch=True)
        self.log("test_precision", precision, on_step=False, on_epoch=True)
        self.log("test_recall", recall, on_step=False, on_epoch=True)
        self.log("test_f1", f1_score, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.val_acc, on_step=False, on_epoch=True)
        return logits


def create_model(config: Config, audio_shape: tuple):
    if config.model_type == "yolo":
        yolo_network = config.create_network()
        yolo_network.compile(  # type: ignore
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(lr=1e-3),
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
        return AudioModel(input_shape=audio_shape)
    elif config.model_type == "ast_model":
        ast_network = config.create_network()
        return ast_network
    return


def create_train_test_val_split(audio_df: AudioDataset) -> tuple:
    train_size = int(0.8 * len(audio_df))
    val_size = int(0.1 * len(audio_df))
    test_size = len(audio_df) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(  # type: ignore
        audio_df, [train_size, val_size, test_size]
    )
    return train_set, val_set, test_set


def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def create_ast_data_and_model(config: Config):
    audio_conf = {
        "num_mel_bins": 128,
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
    audio_data_set = AudiosetDataset(audio_df, audio_conf)
    train_set, val_set, test_set = create_train_test_val_split(audio_data_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        # collate_fn=custom_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )

    audio_model = ASTModel(
        label_dim=1,
        input_fdim=128,
        input_tdim=config.audio_length,
        imagenet_pretrain=True,
        audioset_pretrain=True,
    )

    return audio_model, train_loader, val_loader, test_loader


def train_ast_model(config: Config):
    audio_model, train_loader, val_loader, test_loader = create_ast_data_and_model(
        config
    )
    # wandb_logger = WandbLogger(project="test_rat_USV", reinit=True)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        # logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu"
    )
    trainer.fit(audio_model, train_loader, val_loader)
    trainer.test(audio_model, test_loader)

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
    run.finish()  # type: ignore


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
    # wandb_logger = WandbLogger(project="test_rat_USV", log_model="all")

    # create model and trainer
    if config.model_type == "resnet":
        audio_model = AudioModel(input_shape=audio_shape)

    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    image_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    # copy a few samples for image prediction logger
    imgs = []
    labels = []
    for image, label in image_loader:
        imgs.append(image)
        labels.append(label)
    samples = (torch.cat(imgs), torch.cat(labels))

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        # logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu"
    )
    # train the model
    trainer.fit(
        model=audio_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(audio_model, test_loader)


if __name__ == "__main__":
    config = Config()
    if config.model_type == "yolo" or config.model_type == "long_yolo":
        train_keras_network(config)
    elif config.model_type == "resnet":
        train_torch_network(config)
    elif config.model_type == "ast_model":
        train_ast_model(config)
