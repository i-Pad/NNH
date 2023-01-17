import ga
import constant as const

import random
import operator
import pandas as pd
import pickle
import copy

import tensorflow as tf
from tensorflow.python.keras import layers, models, losses
from tensorflow.keras import datasets
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import keras
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

# model, string, generation
def cost(m, s):
	filters = []

	for i in range(len(s)):
		if s[i] == '1':
			filters.append(i)

	print(filters)

	w, b = m.layers[n_conv].get_weights()
	filter_num = w.shape[3]
	filter_list = list(range(filter_num))

	for i in filters:
		filter_list.remove(i)
	for i in range(len(filter_list)):
		t = filter_list[i]
		w[:, :, :, t] = 0

	m.layers[n_conv].set_weights([w, b])
	m.layers[n_conv].trainable = False

	acc = m.evaluate(x_val, y_val)[1]
	m.layers[n_conv].set_weights([origin_w, b])

	return acc

def save_(m, s, name):
	filters = []

	for i in range(len(s)):
		if s[i] == '1':
			filters.append(i)

	print(filters)

	w, b = m.layers[n_conv].get_weights()
	filter_num = w.shape[3]
	filter_list = list(range(filter_num))

	for i in filters:
		filter_list.remove(i)
	for i in range(len(filter_list)):
		t = filter_list[i]
		w[:, :, :, t] = 0

	m.layers[n_conv].set_weights([w, b])
	m.save(name)
	print('model is saved. terminating program...')


def get_layer_and_filter(l):
	if l == '0':
		return 1, 64
	elif l == '1':
		return '5', 192
	elif l == '2':
		return 9, 382
	elif l == '3':
		return 11, 256
	elif l == '4':
		return 13, 256

def main():
	conv_layer = input()
	if conv_layer == '0':
		MAX_P = const.MAX_P0
		model_name = 'conv' + conv_layer + 'before.h5'
		save_name = 'conv' + conv_layer + 'after.h5'
	else:
		MAX_P = const.MAX_P
		model_name = 'conv' + str(int(conv_layer) - 1) + 'after.h5'
		save_name = 'conv' + conv_layer + 'after.h5'

	global n_conv
	n_conv, n_filter = get_layer_and_filter(conv_layer)

	# load model and copy weight
	Model = load_model(model_name, compile=False)

	for layer in Model.layers[:]:
		layer.trainable = False

	w, b = Model.layers[n_conv].get_weights()
	global origin_w
	origin_w = copy.deepcopy(w)

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

	Model.compile(
		optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
	)

	#Model.summary()
	#print('for test:', cost(Model, '0000000000000000000000000000000000000000000000000000000000000000', save=False, loaded=False, copied=False))

	fitness = []
	totla_fit = 0.0
	fit_sum = 0.0
	best = 0
	worst = 0
	par1 = 0
	par2 = 0

	generation = 1
	is_same = 0

	# initialize population
	n_pop = 0
	pop_string = []
	while(n_pop < MAX_P):
		active = random.sample(range(n_filter), 16)
		temp_string = ''
		for i in range(n_filter):
			temp_string += '1' if i in active else '0'

		pop_string.append(temp_string)

		n_pop += 1

	population = [(p, cost(Model, p)) for p in pop_string]

	res = []
	avg = []
	p_all = []
	while(1):
		# stop condition: Max generation OR performance isn't improved during MAX_S
		if generation > const.MAX_G:
			break

		# sort descending -> index 0 is the best
		population = sorted(population, key=operator.itemgetter(1), reverse=True)

		# check performance improvemnet
		'''
		if population[0][1] == best:
			is_same += 1
		else:
			is_same = 0
		if is_same > const.MAX_S:
			break
		'''

		best = population[0][1]
		worst = population[MAX_P - 1][1]

		fitness = []
		total_fit = 0.0

		if best == worst:
			for _ in range(MAX_P):
				fitness.append(1)
				total_fit += 1
		else:
			for i in range(MAX_P):
				fitness.append(population[i][1] - worst + (best - worst) / (const.COEF_SEL - 1))
				total_fit += fitness[i]

		children = []
		for _ in range(const.COEF_GEN):
			fit_sum = 0.0
			par1 = 0
			par2 = 0

			# selection (roulette wheel, it can be changed)
			rand_float = random.uniform(0, total_fit)
			while(par1 < MAX_P):
				fit_sum += fitness[par1]
				if fit_sum > rand_float:
					break
				par1 += 1
			if par1 == MAX_P:
				par1 -= 1

			fit_sum = 0.0
			rand_float = random.uniform(0, total_fit)
			while(par2 < MAX_P):
				fit_sum += fitness[par2]
				if fit_sum > rand_float:
					break
				par2 += 1
			if par2 == MAX_P:
				par2 -= 1

			# crossover
			#temp_string = ga.OneCrossover(population[par1][0], population[par2][0])
			temp_string = ga.MultiCrossover(population[par1][0], population[par2][0])
			#temp_string = ga.UniformCrossover(population[par1][0], population[par2][0])

			# mutation
			temp_string = ga.OneMutation(temp_string)
			#temp_string = ga.UniformMutation(temp_string)

			# local optimization
			temp_string = ga.localOptimization(temp_string)

			children.append((temp_string, cost(Model, temp_string)))

		#print('before:', population[MAX_P - 1])
		# replace
		for i in range(const.COEF_GEN):
			#print(children[i])
			population[MAX_P - i - 1]  = children[i]

		#print('target:', children[0])
		#print('after:', population[MAX_P - 1])
		population = sorted(population, key=operator.itemgetter(1), reverse=True)
		best = population[0][1]

		print('generation:', generation, 'best:', best)
		generation += 1
		res.append(best)
		avg.append(sum(pair[1] for pair in population) / MAX_P)
		for i in range(MAX_P):
			p_all.append(population[i][1])

	population = sorted(population, key=operator.itemgetter(1), reverse=True)

	best_name = './conv' + conv_layer + 'best.pickle'
	avg_name = './conv' + conv_layer + 'avg.pickle'
	all_name = './conv' + conv_layer + 'all.pickle'

	df_b = pd.DataFrame(res)
	#df_b.to_csv('conv_layer0_best.csv', index=False)
	df_a = pd.DataFrame(avg)
	#df_a.to_csv('conv_layer0_avg.csv', index=False)
	df_all = pd.DataFrame(p_all)
	with open(best_name, 'wb') as fb:
		pickle.dump(df_b, fb, pickle.HIGHEST_PROTOCOL)
	with open(avg_name, 'wb') as fa:
		pickle.dump(df_a, fa, pickle.HIGHEST_PROTOCOL)
	with open(all_name, 'wb') as fall:
		pickle.dump(df_all, fall, pickle.HIGHEST_PROTOCOL)


	ans1 = population[0][0]# string
	ans2 = population[0][1]# acc
	ans3 = 0# num of '1's
	for i in range(len(ans1)):
		if ans1[i] == '1':
			ans3 += 1
	ans_name = './conv' + conv_layer + 'answer.txt'
	with open(ans_name, 'w') as file:
		file.write(str(ans3) + '\n')
		file.write(ans1 + '\n')
		file.write(str(ans2) + '\n')
		file.close()

	# save
	save_(Model, population[0][0], save_name)

if __name__ == '__main__':
	main()
