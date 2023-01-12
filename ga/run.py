import GA_pruning_paper as model
import ga
import constant

import random
import operator

######################
# should change here #
#     conv layer     #
conv_layer = 0
######################


def cost(m, s, flag):
	filters = []

	for i in range(len(s)):
		if s[i] == '1':
			filters.append(i)

	return model.accuracy2(m, filters, conv_layer, flag) if flag == 1 else model.accuracy2(m, filters, conv_layer, flag)[1]

def main():
	Model = model.load_()

	#Model.summary()

	# initialize population

	######################
	# should change here #
	#  number of filters #
	n_filter = 64
	######################

	n_pop = 0
	pop_string = []
	while(n_pop < constant.MAX_P):
		temp_string = ''
		for _ in range(n_filter):
			temp_string += '1' if random.randint(1, 1000) % 2 == 1 else '0'

		pop_string.append(temp_string)

		n_pop += 1

	population = [(p, cost(Model, p, 0)) for p in pop_string]

	fitness = []
	totla_fit = 0.0
	fit_sum = 0.0
	best = 0
	worst = 0
	par1 = 0
	par2 = 0

	generation = 1
	is_same = 0
	while(1):
		# stop condition: Max generation OR performance isn't improved during MAX_S
		if generation > constant.MAX_G:
			break

		# sort descending -> index 0 is the best
		population = sorted(population, key=operator.itemgetter(1), reverse=True)

		# check performance improvemnet
		if population[0][1] == best:
			is_same += 1
		else:
			is_smae = 0

		if is_same > constant.MAX_S:
			break

		best = population[0][1]
		worst = population[constant.MAX_P - 1][1]

		fitness = []
		total_fit = 0.0

		if best == worst:
			for _ in range(constant.MAX_P):
				fitness.append(1)
				totla_fit += 1
		else:
			for i in range(constant.MAX_P):
				fitness.append(population[i][1] - worst + (best - worst) / (constant.COEF_SEL - 1))
				total_fit += fitness[i]

		children = []
		for _ in range(constant.COEF_GEN):
			fit_sum = 0.0
			par1 = 0
			par2 = 0

			# selection (roulette wheel, it can be changed)
			rand_float = random.uniform(0, total_fit)
			while(par1 < constant.MAX_P):
				fit_sum += fitness[par1]
				if fit_sum > rand_float:
					break
				par1 += 1
			if par1 == constant.MAX_P:
				par1 -= 1


			fit_sum = 0.0
			rand_float = random.uniform(0, total_fit)
			while(par2 < constant.MAX_P):
				fit_sum += fitness[par2]
				if fit_sum > rand_float:
					break
				par2 += 1
			if par2 == constant.MAX_P:
				par2 -= 1

			# crossover
			#temp_string = ga.OneCrossover(population[par1][0], population[par2][0])
			#temp_string = ga.MultiCrossover(population[par1][0], population[par2][0])
			temp_string = ga.UniformCrossover(population[par1][0], population[par2][0])

			# mutation
			#if random.uniform(0, 1) < constant.MUTATION_RATE: 
				#temp_string = ga.OneMutation(temp_string)
			temp_string = ga.UniformMutation(temp_string)

			children.append((temp_string, cost(Model, temp_string)))

		# replace
		for i in range(constant.COEF_GEN):
			population[constant.MAX_P - i - 1]  = children[i]

		generation += 1
		print('generation: ', generation, 'best: ', best)

		#model.save


if __name__ == '__main__':
	main()
