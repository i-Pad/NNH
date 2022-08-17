import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import sleep
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Mlight(nn.Module):
	def __init__(self):
		super(Mlight, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
		#self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
		#self.maxp1 = nn.MaxPool1d(3, stride=1, padding=1)
		self.dense1 = nn.Linear(16, 4)

	def forward(self, X):
		#print('before start:', X.shape)
		X = F.relu(self.conv1(X))
		#print('after conv1:', X.shape)
		#X = F.relu(self.conv2(X))
		#print('after conv2:', X.shape)
		#X = self.maxp1(X)
		#print('after pooling:', X.shape)
		X = F.relu(self.conv3(X))
		#print('after conv3:', X.shape)
		#X = self.maxp1(X)
		#print('after pooling:', X.shape)
		X = torch.flatten(X)
		X = X.view(-1, 16)
		#print('after flatten:', X.shape)
		X = self.dense1(X)
		#print('after dense:', X.shape)

		return X

def main():
	n_train = 100000
	X_train = np.zeros((n_train, 4))
	for i in range(n_train):
		X_train[i, :] = np.random.permutation(50)[0: 4]

	X_train = X_train.reshape(n_train, 4, 1)
	y_train = np.sort(X_train, axis=1).reshape(n_train, 4, )

	X_train = X_train.astype(np.float32)
	y_train = y_train.astype(np.float32)

	X_train = torch.from_numpy(X_train)
	y_train = torch.from_numpy(y_train)

	model = Mlight()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device name:', device)
	model.to(device)
	X_train = X_train.to(device)
	y_train = y_train.to(device)

	sample_data = torch.randn(1, 4, 1)
	sample_data = sample_data.to(device)
	output = model(sample_data)
	print(output)

	dataset = TensorDataset(X_train, y_train)

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

				cost = F.mse_loss(pred, y_t)

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
	plt.savefig('more_light.png')
	plt.show()

	torch.save(model, 'more_light.pt')

if __name__ == '__main__':
	main()
