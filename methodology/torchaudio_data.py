# Deep Learning framework
import os
from dataclasses import dataclass
from email.mime import audio
from typing import List

import numpy as np
import pandas as pd

# Pre-trained image models
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Audio processing
import torchaudio
import torchaudio.transforms as T
import tqdm
from config import Config
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset


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

        # Add any preprocessing you like here
        # (e.g., noise removal, etc.)
        ...

        # Add any data augmentations for waveform you like here
        # (e.g., noise injection, shifting time, changing speed and pitch)
        ...

        # Convert to Mel spectrogram
        melspectrogram = T.MelSpectrogram(
            sample_rate=self.target_sample_rate, n_mels=self.n_mfcc, hop_length=512
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


class AudioModel(nn.Module):
    def __init__(
        self, model_name="tf_efficientnet_b3.ns_jft_in1k", pretrained=True, num_classes=1
    ):
        super(AudioModel, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        self.in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(nn.Linear(self.in_features, num_classes))

    def forward(self, images):
        # return model output trough sigmoid
        logits = self.model(images)
        logits = F.sigmoid(logits)
        return logits


def train_one_epoch(model, criterion, optimizer, data_loader, scheduler, epoch, device):
    model.train()
    loss_list = []
    acc_list = []
    for i, data in enumerate(tqdm.tqdm(data_loader)):
        optimizer.zero_grad()
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_list.append(loss.item())
        acc_list.append((outputs.argmax(1) == labels).float().mean().item())
    print(f'epoch: {epoch}: acc:',np.mean(acc_list),'loss: ',np.mean(loss_list))


def run(config: Config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_df = create_df_from_audio_filepaths(config)
    audio_df = AudioDataset(audio_df, 120)
    train_set, val_set = torch.utils.data.random_split(audio_df, [5, 1])  # type: ignore
    audio_model = AudioModel()
    optimizer = optim.Adam(audio_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=60, eta_min=0.00001
    )
    criterion = nn.BCELoss()

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    for epoch in tqdm.tnrange(config.epochs):
        train_one_epoch(audio_model, criterion, optimizer, train_loader, scheduler, epoch, device)



if __name__ == "__main__":
    config = Config()
    run(config)
