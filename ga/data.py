from tensorflow.python.keras import layers, models, losses
import numpy as np
import keras
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer

def load_data():
	train_raw = loadmat("./data/train_32x32.mat")
	test_raw = loadmat("./data/test_32x32.mat")

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

	return train_images, train_labels
