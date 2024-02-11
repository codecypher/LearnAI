#!/usr/bin/env python3
"""
  keras_transformer.py
  This program shows how to perform timeseries classification from scratch.

  We start from raw CSV timeseries files on disk and demonstrate the workflow
  on the FordA dataset from the UCR/UEA archive.

  [Timeseries classification from scratch](https://keras.io)

  [Timeseries classification with a Transformer model](https://keras.io)
  [Timeseries forecasting for weather prediction](https://keras.io)
  [Time series forecasting](https://www.tensorflow.org/tutorials)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


class Params:
    """
    Model hyper-parameters
    """
    def __init__(self):
        self.data_dir = "../data"
        self.root_dir = "../stock_tft"

        self.epochs = 200
        self.batch_size = 64
        self.validation_split = 0.2

        self.input_shape = None
        self.num_classes = None

        self.head_size = 256
        self.num_heads = 4
        self.ff_dim = 4
        self.num_transformer_blocks = 4
        self.mlp_units = [128]
        self.mlp_dropout = 0.4
        self.dropout = 0.25


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    X = data[:, 1:]
    y = data[:, 0]
    print('(X, y):', X.shape, y.shape)
    return X, y.astype(int)


def get_model(model, model_params):
    """
    We can also easily switch between models.
    """
    models = {
        "cnn": create_cnn_model,
        "transformer": create_transformer_model
    }
    return models.get(model.lower())(model_params)


def load_data():
    """
    We will use the FordA_TRAIN file for training and the FordA_TEST file for testing.
    The first column corresponds to the label.
    """
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    X_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    X_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    return (X_train, y_train), (X_test, y_test)


def visualize_data(X_train, y_train, X_test, y_test):
    """
    We visualize one timeseries example for each class in the dataset.
    """
    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    plt.figure()
    for c in classes:
        c_x_train = X_train[y_train == c]
        plt.plot(c_x_train[0], label="class " + str(c))

    plt.legend(loc="best")
    plt.show()
    plt.close()


def data_prep(X_train, y_train, X_test, y_test):
    """
    Data Preprocessing
    """
    # The timeseries data used here are univariate which means that
    # we only have one channel per timeseries example.
    # We will transform the timeseries into a multivariate one with one channel
    # using a simple reshaping via numpy which will allow us to construct a model
    # that is easily applicable to multivariate time series.
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # We shuffle the training set because we will be using the
    # validation_split option later when training.
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Standardize the labels to positive integers (0 and 1).
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return (X_train, y_train), (X_test, y_test)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    The model processes a tensor of shape (batch size, sequence length, features)
    where sequence length is the number of time steps and features is each input timeseries.

    We include residual connections, layer normalization, and dropout.
    The resulting layer can be stacked multiple times.
    """
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def create_transformer_model(params):
    """
    We can stack multiple of those transformer_encoder blocks and we can also
    add the final Multi-Layer Perceptron classification head.

    We also need to reduce the output tensor of the TransformerEncoder part of our model
    to a vector of features for each data point in the current batch.

    A common way to achieve this is to use a pooling layer.
    Here, a GlobalAveragePooling1D layer is sufficient.
    """
    inputs = keras.Input(shape=params.input_shape)
    x = inputs
    for _ in range(params.num_transformer_blocks):
        x = transformer_encoder(x,
                                params.head_size,
                                params.num_heads,
                                params.ff_dim,
                                params.dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in params.mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(params.mlp_dropout)(x)

    outputs = layers.Dense(params.num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def create_cnn_model(params):
    """
    Build the model
    We build a Fully Convolutional Neural Network originally proposed in this paper.
    The implementation is based on the TF 2 version provided here.
    The following hyperparameters (kernel_size, filters, BatchNorm) were found via
    random search using KerasTuner.
    """
    input_layer = layers.Input(params.input_shape)

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)

    gap = layers.GlobalAveragePooling1D()(conv3)

    output_layer = layers.Dense(params.num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train(params, X_train, y_train):
    """
    Train the model
    """
    # To use sparse_categorical_crossentropy, we need to count the number of classes.
    params.num_classes = len(np.unique(y_train))

    params.input_shape = X_train.shape[1:]

    # Create the model
    model = get_model("transformer", params)

    # Build the model
    # model = make_model(input_shape=X_train.shape[1:], num_classes=num_classes)

    # plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "../output/best_forda_model.h5",
            save_best_only=True,
            monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=params.epochs,
        batch_size=params.batch_size,
        validation_split=params.validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate(X_test, y_test):
    """
    Evaluate model on test data
    """
    model = keras.models.load_model("../output/best_forda_model.h5")

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)


def plot_loss(history):
    """
    Plot the model training and validation loss
    """
    metric = "sparse_categorical_accuracy"
    fig = plt.figure(figsize=(10,8))
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")

    # plt.show()
    # plt.close()

    fig.savefig("../output/forda_loss.png")


def main():
    params = Params()
    (X_train, y_train), (X_test, y_test) = load_data()
    (X_train, y_train), (X_test, y_test) = data_prep(X_train, y_train, X_test, y_test)
    history = train(params, X_train, y_train)
    evaluate(X_test, y_test)
    plot_loss(history)


# The driver function (confirm that code is under main function)
if __name__ == "__main__":
    # (X_train, y_train), (X_test, y_test) = load_data()
    # visualize_data(X_train, y_train, X_test, y_test)

    main()

    print("Done!")
