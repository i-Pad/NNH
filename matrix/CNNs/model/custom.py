import torch
import torch.nn as nn
import numpy as np

class multiply(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, X):
		X = X.view(-1, X.shape[1] * X.shape[2], 1)
		X = X.T

		num = X.shape[2]
		ele = X.shape[1]

		for i in range(ele):
			for j in range(i, ele):
				if i == j:
					mid = torch.mul(X[0][i], X[0][j]).view(-1, num, 1)
				else:
					mid = torch.cat([mid, torch.mul(X[0][i], X[0][j]).view(-1, num, 1)])
			if i == 0:
				output = mid.clone().detach()
			else:
				output = torch.cat([output, mid])

		output = output.T
		size = int(ele * (ele + 1) / 2)
		output = output.view(-1, size, 1)

		return output

def main():
	t = torch.FloatTensor([[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]])
	print(t)
	#t = torch.FloatTensor([[[1, 2], [3, 4]], [[2, 3], [4, 5]]])
	#t = torch.FloatTensor([[[1, 2], [3, 4]]])
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device name:', device)
	t = t.to(device)
	print(t.shape)
	t = t.view(-1, t.shape[1] * t.shape[2], 1)

	t = t.T
	#print(t)	

	print('********************')

	print(t[0][0])
	print(t[0][1])

	num = t.shape[2]
	ele = t.shape[1]

	for i in range(ele):
		for j in range(i, ele):
			#temp = torch.mul(t[0][i], t[0][j])
			#print(temp)
			if i == j:
				mid = torch.mul(t[0][i], t[0][j]).view(-1, num, 1)
			else:
				mid = torch.cat([mid, torch.mul(t[0][i], t[0][j]).view(-1, num, 1)])
			#print('mid:', mid)
		if i == 0:
			output = mid.clone().detach()
		else:
			output = torch.cat([output, mid])

	print(output)
	print(output.shape)
	output = output.T
	print(output)
	print(output.shape)

	size = int(ele * (ele + 1) / 2)
	output = output.view(-1, size, 1)
	print(output.T)
	print(output)
	print(output.shape)

if __name__ == '__main__':
	main()
