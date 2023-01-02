import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
		super(MyConv2d, self).__init__()

		self.kernel_size = (kernel_size, kernel_size)
		self.kernel_size_number = kernel_size * kernel_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = (stride, stride)
		self.padding = (padding, padding)
		self.dilation = (dilation, dilation)
		self.weights = nn.Parameters(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size_number)).data.uniform_(0, 1)

def main():
	print('everything is good')

if __name__=='__main__':
	main()
