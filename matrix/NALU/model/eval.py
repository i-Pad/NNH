import torch
#import pytorch_model_summary as summary

from model import NALU
from model import Inverse
from custom import multiply

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	nalu = torch.load('det_with_nalu_and_linear.pt')
	cnalu = torch.load('det_with_custom_and_nalu.pt')
	nalu_input = torch.randint(-10, 11, (1, 4))
	cnalu_input = nalu_input.reshape(-1, 2, 2)
	nalu_input = nalu_input.type(torch.float)
	cnalu_input = cnalu_input.type(torch.float)
	nalu_input = nalu_input.to(device)
	cnalu_input = cnalu_input.to(device)

	npred = nalu(nalu_input)
	cpred = cnalu(cnalu_input)

	print('input')
	print(cnalu_input)
	print('nalu + linear:')
	print(npred)
	print('enconding + nalu:')
	print(cpred)
	print('answer')
	print(cnalu_input.det())

if __name__ == '__main__':
	main()
