import ga
import constant as const
import data

import random
import operator
import pandas as pd
import pickle
import copy

from sys import argv

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

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

def get_name(conv_layer):
	if conv_layer == '0':
		MAX_P = const.MAX_P0
		model_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'before.h5'
		save_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'after.h5'
	else:
		MAX_P = const.MAX_P
		model_name = './result/AlexNet/conv' + str(int(conv_layer) - 1) + '/conv' + str(int(conv_layer) - 1) + 'after.h5'
		save_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'after.h5'
	
	return MAX_P, model_name, save_name

def main():
	conv_layer = argv[1]

	MAX_P, model_name, save_name = get_name(conv_layer)

	global n_conv
	n_conv, n_filter = get_layer_and_filter(conv_layer)

	# load model and copy weight
	model = load_model(model_name, compile=False)

	for layer in model.layers[:]:
		layer.trainable = False

	w, b = model.layers[n_conv].get_weights()
	global origin_w
	origin_w = copy.deepcopy(w)

	# load data
	train_images, train_labels = data.load_data()

	global x_val
	global y_val

	_, x_val, _, y_val = train_test_split(
		train_images, train_labels, test_size=0.15, random_state=22
	)

	# Data augmentation
	datagen = ImageDataGenerator(
		rotation_range=8,
		zoom_range=[0.95, 1.05],
		height_shift_range=0.10,
		shear_range=0.15,
	)

	model.compile(
		optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
	)

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

	population = [(p, cost(model, p)) for p in pop_string]

	res = []
	avg = []
	p_all = []
	while(1):
		# stop condition: Max generation OR no performance improvement during MAX_S
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

			children.append((temp_string, cost(model, temp_string)))

		# replace
		for i in range(const.COEF_GEN):
			population[MAX_P - i - 1]  = children[i]

		population = sorted(population, key=operator.itemgetter(1), reverse=True)
		best = population[0][1]

		print('generation:', generation, 'best:', best)
		generation += 1
		res.append(best)
		avg.append(sum(pair[1] for pair in population) / MAX_P)
		for i in range(MAX_P):
			p_all.append(population[i][1])

	##### save result #####
	population = sorted(population, key=operator.itemgetter(1), reverse=True)

	best_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'best.pickle'
	avg_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'avg.pickle'
	all_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'all.pickle'

	df_b = pd.DataFrame(res)
	df_a = pd.DataFrame(avg)
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

	ans_name = './result/AlexNet/conv' + conv_layer + '/conv' + conv_layer + 'answer.txt'
	with open(ans_name, 'w') as file:
		file.write(str(ans3) + '\n')
		file.write(ans1 + '\n')
		file.write(str(ans2) + '\n')
		file.close()

	save_(model, population[0][0], save_name)

if __name__ == '__main__':
	main()
