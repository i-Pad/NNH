import GA_pruning_paper as model
import ga
import constant as const

import random
import operator
import pandas as pd

######################
# should change here #
#     conv layer     #
conv_layer = 0
######################


def cost(m, s, save, loaded, copied):
	filters = []

	for i in range(len(s)):
		if s[i] == '1':
			filters.append(i)

	print(filters)

	return model.accuracy2(m, filters, conv_layer, save, loaded, copied)

def main():
	Model = model.load_checkpoint_('model3_checkpoint.h5')

	#Model.summary()

	print('for test:', cost(Model, '0000000000000000000000000000000000000000000000000000000000000000', save=False, loaded=False, copied=False))

	# initialize population

	######################
	# should change here #
	#  number of filters #
	n_filter = 64
	######################

	n_pop = 0
	pop_string = []
	while(n_pop < const.MAX_P):
		temp_string = ''
		for _ in range(n_filter):
			temp_string += '1' if random.randint(1, 1000) % 2 == 1 else '0'

		pop_string.append(temp_string)

		n_pop += 1

	population = [(p, cost(Model, p, save=False, loaded=True, copied=True)) for p in pop_string]

	fitness = []
	totla_fit = 0.0
	fit_sum = 0.0
	best = 0
	worst = 0
	par1 = 0
	par2 = 0

	generation = 1
	is_same = 0

	res = []
	avg = []
	while(1):
		# stop condition: Max generation OR performance isn't improved during MAX_S
		if generation > const.MAX_G:
			break

		# sort descending -> index 0 is the best
		population = sorted(population, key=operator.itemgetter(1), reverse=True)

		# check performance improvemnet
		if population[0][1] == best:
			is_same += 1
		else:
			is_same = 0

		'''
		if is_same > const.MAX_S:
			break
		'''

		best = population[0][1]
		worst = population[const.MAX_P - 1][1]

		fitness = []
		total_fit = 0.0

		if best == worst:
			for _ in range(const.MAX_P):
				fitness.append(1)
				total_fit += 1
		else:
			for i in range(const.MAX_P):
				fitness.append(population[i][1] - worst + (best - worst) / (const.COEF_SEL - 1))
				total_fit += fitness[i]

		children = []
		for _ in range(const.COEF_GEN):
			fit_sum = 0.0
			par1 = 0
			par2 = 0

			# selection (roulette wheel, it can be changed)
			rand_float = random.uniform(0, total_fit)
			while(par1 < const.MAX_P):
				fit_sum += fitness[par1]
				if fit_sum > rand_float:
					break
				par1 += 1
			if par1 == const.MAX_P:
				par1 -= 1

			fit_sum = 0.0
			rand_float = random.uniform(0, total_fit)
			while(par2 < const.MAX_P):
				fit_sum += fitness[par2]
				if fit_sum > rand_float:
					break
				par2 += 1
			if par2 == const.MAX_P:
				par2 -= 1

			# crossover
			#temp_string = ga.OneCrossover(population[par1][0], population[par2][0])
			temp_string = ga.MultiCrossover(population[par1][0], population[par2][0])
			#temp_string = ga.UniformCrossover(population[par1][0], population[par2][0])

			# mutation
			#if random.uniform(0, 1) < const.MUTATION_RATE: 
				#temp_string = ga.OneMutation(temp_string)
			temp_string = ga.UniformMutation(temp_string)

			children.append((temp_string, cost(Model, temp_string, save=False, loaded=True, copied=True)))

		#print('before:', population[const.MAX_P - 1])
		# replace
		for i in range(const.COEF_GEN):
			#print(children[i])
			population[const.MAX_P - i - 1]  = children[i]

		generation += 1
		#print('target:', children[0])
		#print('after:', population[const.MAX_P - 1])
		print('generation:', generation, 'best:', best)
		res.append(best)
		avg.append(sum(pair[1] for pair in population) / const.MAX_P)

	population = sorted(population, key=operator.itemgetter(1), reverse=True)

	df_b = pd.DataFrame(res)
	df_b.to_csv('conv_layer0_best.csv', index=False)
	df_a = pd.DataFrame(avg)
	df_a.to_csv('conv_layer0_avg.csv', index=False)
	model.save_(cost(Model, population[0][0], save=True, loaded=True, copied=True), 'conv0-result.h5')

if __name__ == '__main__':
	main()
