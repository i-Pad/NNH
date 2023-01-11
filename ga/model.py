import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers, models, losses
from tensorflow.keras import datasets
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


def save_(n2_model):
    n2_model.save("model2_checkpoint.h5")


def load_():
    n2_model = load_model("model2.h5")
    return n2_model


def load_checkpoint_():
    n2_model = load_model("model2_checkpoint.h5")
    return n2_model


def accuracy2(n2_model, weight_num, num):
    for n2_layer in n2_model.layers[0:23]:
        n2_layer.trainable = False

    w = n2_model.layers[num].get_weights()[0]

    filter_num = n2_model.layers[num].get_weights()[0].shape[3]

    filter_list = list(range(filter_num))

    for i in weight_num:
        filter_list.remove(i)

    for i in range(len(filter_list)):
        t = filter_list[i]
        w[:, :, :, t] = 0

    n2_model.layers[num].trainable = False

    adam = Adam(lr=1e-3)

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_val = x_train[-2000:, :, :, :]
    y_val = y_train[-2000:]
    print("x_train shape" + str(x_train.shape))
    print("y_train shape" + str(y_train.shape))

    sgd = SGD(lr=1e-3, decay=5e-4, momentum=0.9, nesterov=True)

    adam = Adam(lr=1e-3)

    n2_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = n2_model.fit(
        x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val)
    )
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].title.set_text("Training Loss vs Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend(["Train", "Val"])
    axs[1].plot(history.history["accuracy"])
    axs[1].plot(history.history["val_accuracy"])
    axs[1].title.set_text("Training Accuracy vs Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(["Train", "Val"])
    n2_model.evaluate(x_test, y_test)

    return n2_model
