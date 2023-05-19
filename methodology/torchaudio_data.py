import os
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
from lightning.pytorch.loggers import WandbLogger
from tensorflow import keras
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import trange

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
        self.model = nn.Sequential(first_conv_layer, self.model) # type: ignore

    def forward(self, images):
        # return model output trough sigmoid
        logits = self.model(images) # type: ignore
        logits = torch.sigmoid(logits)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.binary_cross_entropy(outputs, labels.unsqueeze(1))
        self.log("test_loss", loss)
        return loss


def create_model(config: Config, audio_shape: tuple):
    if config.model_type == "yolo":
        model = keras.Sequential()
        # create input layer
        model.add(keras.layers.InputLayer(input_shape=list(audio_shape)))
        yolo_network = config.create_network()
        # remove input layer from yolo_network
        yolo_network.layers.pop(0)  # type: ignore
        # create convolutional layer that is compatible with yolo_network
        model.add(keras.layers.Dense(units=300, activation="relu", input_shape=list(audio_shape)))  # type: ignore
        # add yolo network
        model.add(yolo_network)
        # add classification layer
        model.add(keras.layers.Dense(1, activation="sigmoid"))  # type: ignore
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(lr=1e-3),
            metrics=["accuracy"],
        )  # type: ignore
        return model
    elif config.model_type == "resnet":
        return AudioModel(input_shape=audio_shape)
    return


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = keras.losses.BinaryCrossentropy()(targets, model(inputs))
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_keras_network(config: Config):
    audio_df = create_df_from_audio_filepaths(config)
    audio_df = AudioDataset(audio_df, config.audio_length)
    # split dataset
    train_set, val_set = torch.utils.data.random_split(audio_df, [5, 1])  # type: ignore

    # get shape of audio data, for input shape of model
    audio, _ = train_set[0]
    audio_shape = audio.shape

    # create wandb logger
    # wandb_logger = wandb.keras.WandbCallback()
    optimizer_keras = keras.optimizers.Adam(lr=1e-3)

    model = create_model(config, audio_shape)

    ## Note: Rerunning this cell uses the same model parameters

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 10

    for epoch in trange(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()

        # Training loop - using batches of 32
        for x, y in audio_df:
            x, y = x.numpy(), y.numpy()
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer_keras.apply_gradients(zip(grads, model.trainable_variables)) # type: ignore

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True)) # type: ignore

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print(
            "Epoch {}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result()
            )
        )


def train_torch_network(config: Config):
    num_cpus = config.num_cpus
    # create dataset
    audio_df = create_df_from_audio_filepaths(config)
    audio_df = AudioDataset(audio_df, config.audio_length)
    # split dataset
    train_set, val_set = torch.utils.data.random_split(audio_df, [5, 1])  # type: ignore

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
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=num_cpus
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_cpus)

    # train the model
    trainer.fit(audio_model, train_loader, val_loader)

    # stop wandb run


if __name__ == "__main__":
    config = Config()
    if config.model_type == "yolo":
        train_keras_network(config)
    elif config.model_type == "resnet":
        train_torch_network(config)
