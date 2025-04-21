# transfer learning with ResNet50V2

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
import data_pipeline as pipeline

# set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# constants
PROCESSED_DIR = "../data/processed"
IMG_HEIGHT = IMG_WIDTH = 224
NUM_CHANNELS = 3
NUM_CLASSES = 2
EPOCHS = 30
BATCH_SIZE = 32  # smaller batch size for transfer learning

# metrics
metrics = [
    BinaryAccuracy(name="accuracy"),
    Precision(name="precision"),
    Recall(name="recall"),
]


def build_transfer_model(input_shape, dropout_rate=0.3):
    """
    build transfer learning model using resnet50v2

    params
    ------
    input_shape: tuple
        shape of input images (height, width, channels)
    dropout_rate: float
        dropout rate for regularization

    returns
    -------
    model: tf.keras.Model
        compiled keras model
    """
    # load base model
    base_model = ResNet50V2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # freeze base model layers
    base_model.trainable = False

    # build model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    return model


def train_model(model, train_data, val_data, epochs=EPOCHS, learning_rate=0.001):
    """
    train transfer learning model

    params
    ------
    model: tf.keras.Model
        model to train
    train_data: tf.data.Dataset
        training data generator
    val_data: tf.data.Dataset
        validation data generator
    epochs: int
        number of epochs to train
    learning_rate: float
        learning rate for optimizer

    returns
    -------
    history: tf.keras.callbacks.History
        training history
    """
    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    # early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # train model
    history = model.fit(
        train_data, epochs=epochs, validation_data=val_data, callbacks=[early_stopping]
    )

    return history


def fine_tune_model(model, train_data, val_data, epochs=EPOCHS, learning_rate=1e-5):
    """
    fine-tune the entire model

    params
    ------
    model: tf.keras.Model
        model to fine-tune
    train_data: tf.data.Dataset
        training data generator
    val_data: tf.data.Dataset
        validation data generator
    epochs: int
        number of epochs to fine-tune
    learning_rate: float
        learning rate for fine-tuning

    returns
    -------
    history: tf.keras.callbacks.History
        fine-tuning history
    """
    # unfreeze all layers
    model.trainable = True

    # compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    # early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # fine-tune model
    history = model.fit(
        train_data, epochs=epochs, validation_data=val_data, callbacks=[early_stopping]
    )

    return history


def plot_training_history(history, title):
    """
    plot training and validation metrics

    params
    ------
    history: tf.keras.callbacks.History
        training history
    title: str
        plot title
    """
    metrics = ["loss", "accuracy", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        axes[idx].plot(history.history[metric], label="train")
        if f"val_{metric}" in history.history:
            axes[idx].plot(history.history[f"val_{metric}"], label="val")
        axes[idx].set_title(f"{metric}")
        axes[idx].set_xlabel("epoch")
        axes[idx].set_ylabel(metric)
        axes[idx].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    plot confusion matrix

    params
    ------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted labels
    class_names: list
        list of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.show()
