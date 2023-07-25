import os
import sys

import seaborn as sns
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightning.pytorch.callbacks import EarlyStopping as pl_EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint as pl_ModelCheckpoint
from sklearn.model_selection import train_test_split
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
from module import DataGenerator, F1Metric, PrecisionMetric, RecallMetric
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
        train_test="train",
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
        self.train_test = train_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load audio from file to waveform
        audio, sample_rate = torchaudio.load(self.file_paths[index])  # type: ignore

        # Convert to mono
        audio = torch.mean(audio, dim=0)  # type: ignore

        # Resample
        if sample_rate != self.target_sample_rate:
            resample = T.Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)

        # apply data augmentation, only for training
        if self.train_test == "train":
            audio = self.augment(audio, self.target_sample_rate)

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

            return (
                torch.stack([spec]).detach(),
                torch.tensor(self.labels[index]).float(),
            )

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

        return torch.stack([mfcc]).detach(), torch.tensor(self.labels[index]).float()

    def select_augment_strategy(self):
        """Select augmentation strategy from a list of options.

        Returns:
            np.ndarray: array of selected options
        """
        options = ["add_noise", "time_stretch", "pitch_shift"]
        # from these options, randomly select none to all
        num_options = np.random.randint(0, len(options) + 1)
        # randomly select options
        selected_options = np.random.choice(options, num_options, replace=False)
        return selected_options

    def augment(self, audio, sample_rate):
        """Apply augmentation to audio and return augmented audio"""

        # select augmentation strategy
        selected_options = self.select_augment_strategy()
        # apply augmentation
        for option in selected_options:
            if option == "add_noise":
                # select a random sigma from acceptable range: [0.001, 0.015]
                sigma = np.random.uniform(0.001, 0.015)
                noise = sigma * torch.randn_like(audio)
                audio = audio + noise
            elif option == "time_stretch":
                # select rate from acceptable range: [0.8, 1.25]
                rate = np.random.uniform(0.8, 1.25)
                # convert audio to numpy array
                audio = audio.detach().numpy()
                audio = librosa.effects.time_stretch(audio, rate=rate)
                # convert back to torch tensor
                audio = torch.from_numpy(audio)
            elif option == "pitch_shift":
                # select n_steps from -4 to 4, where 0 is no change
                n_steps = np.random.randint(-4, 5)
                while n_steps == 0:
                    n_steps = np.random.randint(-4, 5)
                audio = torchaudio.transforms.PitchShift(
                    sample_rate=sample_rate, n_steps=n_steps
                )(audio)
                audio.detach()
        return audio


class AudioModel(pl.LightningModule):
    def __init__(
        self, config: Config, num_classes: int, input_shape, load_weights: bool = True
    ):
        super().__init__()
        self.model = config.create_network(load_weights=load_weights)
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
                "val_auc_roc": BinaryAUROC(),
                "val_precision": torchmetrics.Precision(task="binary"),
                "val_recall": torchmetrics.Recall(task="binary"),
            }
        )

        self.metric_collection_test = torchmetrics.MetricCollection(
            {
                "test_accuracy": torchmetrics.Accuracy(task="binary"),
                "test_f1": BinaryF1Score(),
                "test_auc_roc": BinaryAUROC(),
                "test_precision": torchmetrics.Precision(task="binary"),
                "test_recall": torchmetrics.Recall(task="binary"),
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
        logits_int = logits.round().long()
        self.metric_collection_val.update(logits, y.unsqueeze(1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.metric_collection_val, on_step=False, on_epoch=True, prog_bar=True  # type: ignore
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y.unsqueeze(1))
        logits_int = logits.round().long()
        self.metric_collection_test.update(logits_int, y.unsqueeze(1))
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            self.metric_collection_test, on_step=False, on_epoch=True, prog_bar=True  # type: ignore
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


def create_model(config: Config, audio_shape: list, load_weights: bool = True):
    if config.model_type == "yolo" or "long_yolo":
        yolo_network = config.create_network(
            shape=audio_shape, load_weights=load_weights
        )
        yolo_network.compile(  # type: ignore
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(
                    name="auc_roc", curve="ROC", num_thresholds=200, multi_label=False
                ),
                F1Metric(name="f1"),
                PrecisionMetric(name="precision"),
                RecallMetric(name="recall"),
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


def train_keras_network(
    config: Config,
    train_audio_df: pd.DataFrame,
    test_audio_df: pd.DataFrame,
    load_weights: bool = True,
) -> np.ndarray:
    """function for training YOLO and long YOLO pretrained models

    Args:
        config (Config): experiment setyp
        train_audio_df (pd.DataFrame): training dataframe
        test_audio_df (pd.DataFrame): test dataframe
        load_weights (bool, optional): whether to load pretrained weights. Defaults to True.

    Returns:
        np.ndarray: model predictions
    """
    audio_df = AudioDataset(
        train_audio_df,
        config.audio_length,
        spectrogram=config.spectrogram,
        delta=config.delta,
        delta_delta=config.delta_delta,
        n_mfcc=config.n_mfcc,
        target_sample_rate=config.target_sample_rate,
        train_test="train",
    )
    test_audio_set = AudioDataset(
        test_audio_df,
        config.audio_length,
        spectrogram=config.spectrogram,
        delta=config.delta,
        delta_delta=config.delta_delta,
        n_mfcc=config.n_mfcc,
        target_sample_rate=config.target_sample_rate,
        train_test="test",
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
        project="Rat_classification_5",
        reinit=True,
        name=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
        config=config.__dict__,
    )
    callbacks = [
        WandbCallback(save_model=False),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1),
        ModelCheckpoint(
            filepath=f"models/{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}.hdf5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            verbose=1,
            min_delta=0,
            restore_best_weights=True,
        ),
    ]

    model = create_model(config, audio_shape, load_weights=load_weights)
    # convert torch loader to keras compatible object
    train_loader = DataGenerator(train_loader, 2)
    val_loader = DataGenerator(val_loader, 2)
    test_loader = DataGenerator(test_loader, 2)
    model.fit(train_loader, epochs=config.epochs, validation_data=val_loader, callbacks=callbacks)  # type: ignore
    test_stats = model.evaluate(test_loader, callbacks=WandbCallback())  # type: ignore
    model_predictions = model.predict(test_loader)  # type: ignore
    # get model metrics
    metrics = {}
    for i, metric in enumerate(model.metrics_names):
        metrics[f"test_{metric}"] = test_stats[i]
    wandb.log(metrics)

    run.finish()  # type: ignore
    return model_predictions


def train_torch_network(
    config: Config,
    train_audio_df: pd.DataFrame,
    test_audio_df: pd.DataFrame,
    load_weights: bool = True,
) -> np.ndarray:
    """function for training resnet model

    Args:
        config (Config): experiment setup
        train_audio_df (pd.DataFrame): training dataframe
        test_audio_df (pd.DataFrame): test dataframe
        load_weights (bool, optional): Whether to use pretraining or not. Defaults to True.

    Returns:
        np.ndarray: model predictions
    """
    num_cpus = config.num_cpus
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    # create dataset
    audio_df = AudioDataset(
        train_audio_df,
        config.audio_length,
        delta=config.delta,
        delta_delta=config.delta_delta,
        target_sample_rate=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
        train_test="train",
    )
    test_audio_set = AudioDataset(
        test_audio_df,
        config.audio_length,
        delta=config.delta,
        delta_delta=config.delta_delta,
        target_sample_rate=config.target_sample_rate,
        n_mfcc=config.n_mfcc,
        train_test="test",
    )
    # split dataset
    train_set, val_set = create_train_val_split(audio_df)

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = audio.shape

    # create wandb logger
    wandb_logger = WandbLogger(
        project="Rat_classification_5",
        log_model=True,
        name=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
    )

    # create modelcheckpoint callback
    checkpoint_callback = pl_ModelCheckpoint(
        dirpath=f"models/",
        filename=f"{config.model_type}_spec_{config.spectrogram}_D_{config.delta}_DD_{config.delta_delta}",
        monitor="val_loss",
        verbose=True,
        save_top_k=1,
        mode="min",
    )
    # initialize the early stopping callback
    early_stop_callback = pl_EarlyStopping(
        monitor="val_loss",
        patience=5,
        strict=False,
        verbose=False,
        mode="min",
        min_delta=0.00,
    )
    # create model and trainer
    audio_model = AudioModel(
        input_shape=audio_shape, config=config, num_classes=1, load_weights=load_weights
    )

    # create dataloaders
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=num_cpus
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_cpus)
    test_loader = DataLoader(
        test_audio_set, batch_size=1, shuffle=False, num_workers=num_cpus
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    # train the model
    trainer.fit(
        model=audio_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(audio_model, test_loader)
    # get predictions from model on test set
    model_predictions = trainer.predict(audio_model, test_loader)
    model_predictions = torch.cat(model_predictions).numpy()  # type: ignore
    wandb_logger.experiment.finish()
    return model_predictions


def experiments() -> list:
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

    experiment_list = [
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
    ]
    experiment_list1 = [
        resnet_spec,
        resnet_mfcc,
        resnet_mfcc_delta,
        resnet_mfcc_delta_delta,
    ]
    return experiment_list


def final_experiment_no_transfer():
    experiment = Config(model_type="yolo", spectrogram=False, delta=True)
    fix_gpu()
    train_audio = pd.read_csv("train_audio.csv")
    test_audio = pd.read_csv("test_audio.csv")
    predictions = train_keras_network(
        experiment,
        train_audio_df=train_audio,
        test_audio_df=test_audio,
        load_weights=True,
    )
    # create confusion matrix
    predictions = predictions.round().astype(int)
    test_audio["predictions"] = predictions
    # create a nice seaborn plot of the confusion matrix
    confusion_matrix = pd.crosstab(
        test_audio["label"],
        test_audio["predictions"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    plt.figure(figsize=(10, 7))
    plt.savefig("confusion_matrix1.png")
    plt.show()


def load_model_and_create_confusion_matrix():
    experiment = Config(model_type="yolo", spectrogram=False, delta=True)
    fix_gpu()
    train_audio = pd.read_csv("train_audio.csv")
    test_audio = pd.read_csv("test_audio.csv")

    model_predictions = train_keras_network(
        experiment,
        train_audio_df=train_audio,
        test_audio_df=test_audio,
        load_weights=True,
    )
    # create confusion matrix
    model_predictions = model_predictions.round().astype(int)
    test_audio["predictions"] = model_predictions
    # create a nice seaborn plot of the confusion matrix
    confusion_matrix = pd.crosstab(
        test_audio["label"],
        test_audio["predictions"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    plt.figure(figsize=(10, 7))
    plt.title("Confusion matrix")
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    # save confusion matrix
    plt.savefig("confusion_matrix1.png")
    plt.show()


if __name__ == "__main__":
    #load_model_and_create_confusion_matrix()
    config = Config()
    fix_gpu()
    experiments = experiments()  # type: ignore
    # split audio dataframe into train and test
    train_audio = pd.read_csv("train_audio.csv")
    test_audio = pd.read_csv("test_audio.csv")
    for experiment in experiments:  # type: ignore
        predictions = None
        # if experiment model type contains yolo, train yolo model
        if "yolo" in experiment.model_type:
            predictions = train_keras_network(
                experiment, train_audio_df=train_audio, test_audio_df=test_audio
            )
        # if experiment model type contains resnet, train resnet model
        elif "resnet" in experiment.model_type:
            predictions = train_torch_network(
                experiment, train_audio_df=train_audio, test_audio_df=test_audio
            )
        # save predictions
        np.save(
            f"{experiment.model_type}_spec_{experiment.spectrogram}_D_{experiment.delta}_DD_{experiment.delta_delta}.npy",
            predictions,  # type: ignore
        )
