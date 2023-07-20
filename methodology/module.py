import lightning.pytorch as pl
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K
import torch
from tensorflow.keras.metrics import Metric

import wandb


class DataGenerator(k.utils.Sequence):
    """
    class to be fed into model.fit_generator method of tf.keras model

    uses a pytorch dataloader object to create a new generator object that can be used by tf.keras
    dataloader in pytorch must be used to load image data
    transforms on the input image data can be done with pytorch, model fitting still with tf.keras

    ...

    Attributes
    ----------
    gen : torch.utils.data.dataloader.DataLoader
        pytorch dataloader object; should be able to load image data for pytorch model
    ncl : int
        number of classes of input data; equal to number of outputs of model
    """

    def __init__(self, gen, ncl):
        """
        Parameters
        ----------
        gen : torch.utils.data.dataloader.DataLoader
            pytorch dataloader object; should be able to load image data for pytorch model
        ncl : int
            number of classes of input data; equal to number of outputs of model
        """
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        """
        function used by model.fit_generator to get next input image batch

        Variables
        ---------
        ims : np.ndarray
            image inputs; tensor of (batch_size, height, width, channels); input of model
        lbs : np.ndarray
            labels; tensor of (batch_size, number_of_classes); correct outputs for model
        """
        # catch when no items left in iterator
        try:
            ims, lbs = next(
                self.iter
            )  # generation of data handled by pytorch dataloader
        # catch when no items left in iterator
        except StopIteration:
            self.iter = iter(self.gen)  # reinstanciate iterator of data
            ims, lbs = next(
                self.iter
            )  # generation of data handled by pytorch dataloader
        # swap dimensions of image data to match tf.keras dimension ordering
        ims = np.swapaxes(np.swapaxes(ims.detach().numpy(), 1, 3), 1, 2)
        # convert labels to numpy ints
        lbs = lbs.numpy().astype(int)
        return ims, lbs

    def __len__(self):
        """
        function that returns the number of batches in one epoch
        """
        return len(self.gen)


class RecallMetric(Metric):
    def __init__(self, name='recall', **kwargs):
        super(RecallMetric, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32) # Ensuring y_true is float32
        y_pred = tf.cast(y_pred, tf.float32) # Ensuring y_pred is float32
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        self.recall = recall

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall = 0



class PrecisionMetric(Metric):
    def __init__(self, name="precision", **kwargs):
        super(PrecisionMetric, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32) # Ensuring y_true is float32
        y_pred = tf.cast(y_pred, tf.float32) # Ensuring y_pred is float32
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        self.precision = precision

    def result(self):
        return self.precision

    def reset_state(self):
        self.precision = 0


class F1Metric(Metric):
    def __init__(self, name="f1", **kwargs):
        super(F1Metric, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32) # Ensuring y_true is float32
        y_pred = tf.cast(y_pred, tf.float32) # Ensuring y_pred is float32
        precision = PrecisionMetric()
        recall = RecallMetric()
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)
        self.f1 = 2 * ((precision.result() * recall.result()) / (precision.result() + recall.result() + K.epsilon()))

    def result(self):
        return self.f1

    def reset_state(self):
        self.f1 = 0


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=8):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        trainer.logger.experiment.log(  # type: ignore
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(val_imgs, preds, self.val_labels)
                ],
                "global_step": trainer.global_step,
            }
        )
