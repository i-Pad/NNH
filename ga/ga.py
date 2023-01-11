import random
import constant

def OneCrossover(s1, s2):
	n = random.randint(0, len(s1) - 1)

	return s1[:n] + s2[n:]

def MultiCrossover(s1, s2):
	p = sorted(random.sample(range(len(s1)), constant.POINTS))
	#print(p)

	i = 0
	while(i < constant.POINTS):
		if i == 0:
			s = s1[:p[i]]
			i += 1
			continue

		temp = s2[p[i - 1]:p[i]] if i % 2 == 1 else s1[p[i - 1]:p[i]]
		s += temp
		i += 1

	s += s2[p[i - 1]:] if i % 2 == 1 else s1[p[i - 1]:]

	return s

def UniformCrossover(s1, s2):
	s = ''

	for i in range(len(s1)):
		r = random.uniform(0, 1)
		#print(r)

		s += s1[i] if r > constant.UNIFORM else s2[i]

	return s

def flip(s, p):
	l = list(s)
	l[p] = '1' if l[p] == '0' else '0'

	return ''.join(l)

def OneMutation(s):
	r = random.randint(0, len(s) - 1)
	#print(r)

	return flip(s, r)

def UniformMutation(s):
	for i in range(len(s)):
		r = random.uniform(0, 1)

		if r <= constant.MUTATION_RATE:
			#print('flip at:', i)
			s = flip(s, i)

	return s

def main():
	s1 = 'abcdefgh'
	s2 = 'ijklmnop'
	print(s1)
	print(s2)
	s3 = OneCrossover(s1, s2)
	print(s3)
	s4 = MultiCrossover(s1, s2)
	print(s4)
	s5 = UniformCrossover(s1, s2)
	print(s5)
	s6 = '11110000'
	s7 = OneMutation(s6)
	print(s7)
	s8 = UniformMutation(s6)
	print(s8)


if __name__ == '__main__':
	main()
