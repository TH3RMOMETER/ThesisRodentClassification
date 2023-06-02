from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch
from config import Config
from methodology.torchaudio_data import create_df_from_audio_filepaths, AudioDataset, create_train_test_val_split

config = Config()
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

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

with torch.no_grad():
    logits = model(**train_set).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)

# compute loss - target_label is e.g. "down"
target_label = model.config.id2label[0]
inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
loss = model(**inputs).loss
print(round(loss.item(), 2))