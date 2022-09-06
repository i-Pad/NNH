import torch
import pytorch_model_summary as summary
from inverse import Inverse
from custom import multiply

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	mul = torch.load('inverse_mse_mul_conv.pt')
	sample_input = torch.randint(10, (1, 2, 2))
	sample_input = sample_input.type(torch.float)
	sample_input = sample_input.to(device)
	print(summary.summary(mul, sample_input, show_input=True))

	pred = mul(sample_input)
	pred = pred.reshape(2, 2)
	sample_input = sample_input.reshape(2, 2)
	print('input')
	print(sample_input)
	print('prediction')
	print(pred)
	print('answer')
	print(torch.inverse(sample_input))
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
