import torch
import torch.nn as nn
import numpy as np

from torch import Tensor, exp, log
from torch.nn import Sequential
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.functional import linear
from torch import sigmoid, tanh

'''
paper: Neural Arithmetic Logic Units (https://arxiv.org/pdf/1808.00508.pdf)
'''

class NacCell(nn.Module):
	def __init__(self, in_shape, out_shape):
		super().__init__()
		self.in_shape = in_shape
		self.out_shape = out_shape

		self.W_ = Parameter(Tensor(out_shape, in_shape))
		self.M_ = Parameter(Tensor(out_shape, in_shape))

		xavier_uniform_(self.W_), xavier_uniform_(self.M_)
		self.register_parameter('bias', None)

	# a = Wx
	# W = tanh(W) * sigmoid(M)
	# * is elementwise product
	def forward(self, X):
		#print('W:', self.W_.shape, 'X:', X.shape)
		W = tanh(self.W_) * sigmoid(self.M_)

		# linear: XW^T + b
		return linear(X, W.T, self.bias)
		#return torch.matmul(X, W.T)

class NaluCell(nn.Module):
	def __init__(self, in_shape, out_shape):
		super().__init__()
		self.in_shape = in_shape
		self.out_shape = out_shape

		self.G = Parameter(Tensor(out_shape, in_shape))
		self.nac = NacCell(out_shape, in_shape)

		xavier_uniform_(self.G)
		# epsilon prevents log0
		self.eps = 1e-5
		self.register_parameter('bias', None)

	# y = g * a + (1 - g) * m
	# m = exp W(log(|x| + e)), g = sigmoid(Gx)
	# * is elementwise product
	# a is from nac
	def	forward(self, X):
		a = self.nac(X)
		g = sigmoid(linear(X, self.G, self.bias))

		ag = g * a
		log_in = log(abs(X) + self.eps)
		m = exp(self.nac(log_in))
		md = (1 - g) * m

		return ag + md

class NaluLayer(nn.Module):
	def __init__(self, input_shape, output_shape, n_layers, hidden_shape):
		super().__init__()
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.n_layers = n_layers
		self.hidden_shape = hidden_shape

		#layers = [NaluCell(hidden_shape if n > 0 else input_shape, hidden_shape if n < n_layers - 1 else output_shape) for n in range(n_layers)]
		layers = [NaluCell(hidden_shape if n > 0 else input_shape, hidden_shape if n < n_layers - 1 else output_shape) for n in range(n_layers)]
		self.model = Sequential(*layers)

	def forward(self, X):
		return self.model(X)

def main():
	t = torch.FloatTensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device name:', device)

	t = t.to(device)
	print(t.shape)

	model = NaluLayer(input_shape=4, output_shape=1, n_layers=3, hidden_shape=4)
	model.to(device)

	print(t)

	output = model(t)
	print(output)

	print(model)

if __name__ == '__main__':
	main()
