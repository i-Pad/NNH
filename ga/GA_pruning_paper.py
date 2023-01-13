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
import numpy as np
import keras
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

import copy

def save_(n2_model, f_name):
	n2_model.save(f_name)


def load_(f_name='model3.h5'):
	n2_model = load_model(f_name)
	return n2_model


def load_checkpoint_(f_name='model3_checkpoint.h5'):
	n2_model = load_model(f_name, compile=False)
	return n2_model

def accuracy2(n2_model, weight_num, num, save, loaded, copied):
	for n2_layer in n2_model.layers[:]:
		n2_layer.trainable = False

	w, b= n2_model.layers[num].get_weights()

	if copied == False:
		global old_w
		old_w = copy.deepcopy(w)

	filter_num = n2_model.layers[num].get_weights()[0].shape[3]
	filter_list = list(range(filter_num))

	for i in weight_num:
		filter_list.remove(i)

	for i in range(len(filter_list)):
		t = filter_list[i]
		w[:, :, :, t] = 0

	n2_model.layers[num].set_weights([w, b])
	n2_model.layers[num].trainable = False

	if loaded == False:
		train_raw = loadmat("train_32x32.mat")
		test_raw = loadmat("test_32x32.mat")

		# Load images and labels

		train_images = np.array(train_raw["X"])
		test_images = np.array(test_raw["X"])

		train_labels = train_raw["y"]
		test_labels = test_raw["y"]

		train_images = np.moveaxis(train_images, -1, 0)
		test_images = np.moveaxis(test_images, -1, 0)

		train_images = train_images.astype("float64")
		test_images = test_images.astype("float64")

		# Convert train and test labels into 'int64' type

		train_labels = train_labels.astype("int64")
		test_labels = test_labels.astype("int64")

		train_images /= 255.0
		test_images /= 255.0

		lb = LabelBinarizer()
		train_labels = lb.fit_transform(train_labels)
		test_labels = lb.fit_transform(test_labels)

		global x_val
		global y_val

		x_train, x_val, y_train, y_val = train_test_split(
			train_images, train_labels, test_size=0.15, random_state=22
		)

		# Data augmentation

		datagen = ImageDataGenerator(
			rotation_range=8,
			zoom_range=[0.95, 1.05],
			height_shift_range=0.10,
			shear_range=0.15,
		)

		n2_model.compile(
			optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
		)

	if save == True:
		return n2_model
	else:
		acc = n2_model.evaluate(x_val, y_val)[1]
		n2_model.layers[num].set_weights([old_w, b])

		return acc
