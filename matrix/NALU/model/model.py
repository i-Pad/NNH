import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nalu
import custom

from numpy.linalg import inv, det
from tqdm import tqdm
from time import sleep
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class NALU(nn.Module):
	def __init__(self):
		super(NALU, self).__init__()
		self.nalu1 = nalu.NaluLayer(input_shape=4, output_shape=2, n_layers=2, hidden_shape=4)
		#self.nalu2 = nalu.NaluLayer(input_shape=2, output_shape=1, n_layers=1, hidden_shape=2)
		#self.nalu3 = nalu.NaluLayer(input_shape=2, output_shape=1, n_layers=4, hidden_shape=2)
		self.dense1 = nn.Linear(2, 1)

	def forward(self, X):
		#X = torch.cat([self.nalu1(X), self.nalu2(X)], dim=1)
		X = self.nalu1(X)
		#X = self.nalu3(torch.cat([self.nalu1(X), self.nalu2(X)], dim=1))
		#X = self.nalu2(X)
		X = self.dense1(X)
		X = X.squeeze()

		return X

class Inverse(nn.Module):
	def __init__(self):
		super(Inverse, self).__init__()
		self.custom1 = custom.multiply()
		self.nalu1 = nalu.NaluLayer(10, 1, 1, 10)

	def forward(self, X):
		X = self.custom1(X)
		X = X.squeeze()
		X = self.nalu1(X)

		return X

def main():
	#t = torch.FloatTensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

	num = 1
	n_train = 100000
	X_train = []
	y_train = []

	# data range: [1, 55]
	while(num <= n_train):
		A = np.random.randint(1, 50, size=(2, 2))

		if det(A) == 0:
			continue
		else:
			A = A.astype(np.float32)
			X_train.append(A)
			y_train.append(det(A))
			num += 1

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_train = torch.from_numpy(X_train)
	X_train = X_train.reshape(n_train, 4)
	y_train = torch.from_numpy(y_train)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device name:', device)

	#model = nalu.NaluLayer(input_shape=2, output_shape=1, n_layers=2, hidden_shape=2)
	model = NALU()
	#model = Inverse()

	#t = t.to(device)
	model.to(device)
	X_train = X_train.to(device)
	y_train = y_train.to(device)

	print(X_train.shape)
	print(y_train.shape)

	'''
	C_data = []
	for _ in range(3):
		check_data = np.random.randint(1, 50, size=(2, 2))
		check_data = check_data.astype(np.float32)
		C_data.append(check_data)		

	C_data = np.array(C_data)
	C_data = torch.from_numpy(C_data)
	print(C_data)
	C_data = C_data.reshape(3, 4, 1)
	print(C_data)
	'''

	'''
	output = model(t)
	print(t)
	print(output)
	print(model)
	'''

	dataset = TensorDataset(X_train, y_train)

	b_size = 128
	dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	epochs = 100
	cur_loss = 0.0
	epos = []
	los = []

	for epoch in range(epochs):
		with tqdm(dataloader) as tepoch:
			for batch_idx, samples in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
				X_t, y_t = samples

				pred = model(X_t)

				#print('pred size:', pred.shape)
				#print('y_t size:', y_t.shape)
				pred = pred.squeeze()
				cost = F.mse_loss(pred, y_t)

				optimizer.zero_grad()
				cost.backward()
				optimizer.step()

				tepoch.set_postfix(loss=cost.item())
				cur_loss = cost.item()

			epos.append(epoch + 1)
			los.append(cur_loss)

	plt.plot(epos, los, label='train')
	plt.title('model loss')
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.savefig('get_det_with_nalu_and_linear.png')
	plt.show()

	torch.save(model, 'get_det_with_nalu_and_linear.pt')

if __name__=='__main__':
	main()
