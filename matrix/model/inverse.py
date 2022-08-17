import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from numpy.linalg import inv, det
from tqdm import tqdm
from time import sleep
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Inverse(nn.Module):
	def __init__(self):
		super(Inverse, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
		self.maxp1 = nn.MaxPool1d(3, stride=1, padding=1)
		self.dense1 = nn.Linear(16, 2)

	def forward(self, X):
		#print('before start:', X.shape)
		X = F.relu(self.conv1(X))
		#print('after conv1:', X.shape)
		X = F.relu(self.conv2(X))
		#print('after conv2:', X.shape)
		X = self.maxp1(X)
		#print('after pooling:', X.shape)
		X = F.relu(self.conv3(X))
		#print('after conv3:', X.shape)
		X = self.maxp1(X)
		#print('after pooling:', X.shape)
		X = torch.flatten(X)
		X = X.view(-1, 16)
		#print('after flatten:', X.shape)
		X = self.dense1(X)
		#print('after dense:', X.shape)
		X = X.view(-1, 2, 2)
		#print('final:', X.shape)

		return X

def my_loss(output, target):
	#use determinant
	#print('output:', output)
	#print('target', target)
	loss = torch.bmm(output, target)
	#print('after bmm:', loss)
	loss = torch.linalg.det(loss)
	#print('after det:', loss)
	loss = torch.mean(loss) - 1
	#print('final:', loss)

	return abs(loss)

def main():
	'''
	A = np.random.randint(50, size=(2, 2))
	print(A)
	A = A.astype(np.float32)
	print(A)
	Ainv = inv(A)
	print(Ainv)

	B = [[3, 1], [9, 3]]
	try:
		Binv = inv(B)
		print(Binv)
	except:
		print('no inverse')

	C = np.dot(A, Ainv)
	print(C)
	print(det(C))
	'''

	cur = 1
	n_train = 100000
	X_train = []
	y_train = []

	while(1):
		if cur > n_train:
			break

		A = np.random.randint(50, size=(2, 2))
		if det(A) == 0:
			continue
		else:
			A = A.astype(np.float32)
			X_train.append(A)
			y_train.append(inv(A))
			cur += 1

	det_train = X_train.copy()

	det_train = np.array(det_train)
	X_train = np.array(X_train)
	y_train = np.array(y_train)


	det_train = torch.from_numpy(det_train)
	X_train = torch.from_numpy(X_train)
	y_train = torch.from_numpy(y_train)

	print(X_train.shape)
	print(y_train.shape)

	model = Inverse()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device name:', device)
	model.to(device)
	det_train = det_train.to(device)
	X_train = X_train.to(device)
	y_train = y_train.to(device)


	sample_data = torch.randn(1, 2, 2)
	sample_data = sample_data.to(device)
	output = model(sample_data)
	print(output)

	#dataset = TensorDataset(X_train, y_train)
	dataset = TensorDataset(X_train, det_train)

	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	epochs = 100
	cur_loss = 0.0
	Xs = []
	Ys = []

	for epoch in range(epochs):
		with tqdm(dataloader) as tepoch:
			for batch_idx, samples in enumerate(tepoch):
				tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
				X_t, y_t = samples

				#X_tt = X_t[i].reshape(1, 4, 1)
				pred = model(X_t)

				#cost = F.mse_loss(pred, y_t)
				cost = my_loss(pred, y_t)

				optimizer.zero_grad()
				cost.backward()
				optimizer.step()

				tepoch.set_postfix(loss=cost.item())
				cur_loss = cost.item()
		Xs.append(epoch + 1)
		Ys.append(cur_loss)

	plt.plot(Xs, Ys, label='train')
	plt.title('model loss')
	plt.xlabel('Epoch')
	plt.ylabel('loss')
	plt.legend()
	plt.savefig('inverse_det.png')
	plt.show()

	torch.save(model, 'inverse_det.pt')

if __name__ == '__main__':
	main()
