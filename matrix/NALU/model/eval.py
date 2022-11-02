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
	'''
	origin = torch.load('mse_inverse.pt')
	biloss = torch.load('bi_inverse_det.pt')
	absloss = torch.load('abs_inverse_det.pt')

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	sample_input = torch.randint(10, (1, 2, 2))
	sample_input = sample_input.type(torch.float)
	sample_input = sample_input.to(device)

	print(summary.summary(origin, sample_input, show_input=True))
	print(summary.summary(biloss, sample_input, show_input=True))
	print(summary.summary(absloss, sample_input, show_input=True))

	pred1 = origin(sample_input)
	pred2 = biloss(sample_input)
	pred3 = absloss(sample_input)

	pred1 = pred1.reshape(2, 2)
	pred2 = pred2.reshape(2, 2)
	pred3 = pred3.reshape(2, 2)
	sample_input = sample_input.reshape(2, 2)

	print(sample_input)
	print(pred1)
	print(pred2)
	print(pred3)

	A = torch.mm(sample_input, pred1)
	B = torch.mm(sample_input, pred2)
	C = torch.mm(sample_input, pred3)
	
	print(A)
	print(B)
	print(C)

	print(torch.det(A))
	print(torch.det(B))
	print(torch.det(C))

	print(torch.inverse(sample_input))
	'''

if __name__ == '__main__':
	main()
