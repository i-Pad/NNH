import torch
import pytorch_model_summary as summary
from sort import Origin
from light import Light
from semi import Semi
from more_light import Mlight
from super_light import Slight

def main():
	om = torch.load('origin.pt')
	lm = torch.load('light.pt')
	sem = torch.load('semi.pt')
	mm = torch.load('more_light.pt')
	sm = torch.load('super_light.pt')

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	sample_input = torch.randint(50, 100, (1, 4, 1))
	sample_input = sample_input.type(torch.float)
	sample_input = sample_input.to(device)

	print(summary.summary(om, sample_input, show_input=True))
	print(summary.summary(lm, sample_input, show_input=True))
	print(summary.summary(sem, sample_input, show_input=True))
	print(summary.summary(mm, sample_input, show_input=True))
	print(summary.summary(sm, sample_input, show_input=True))

	pred1 = om(sample_input)
	pred2 = lm(sample_input)
	pred3 = sem(sample_input)
	pred4 = mm(sample_input)
	pred5 = sm(sample_input)

	print(sample_input.reshape(1, 4))
	print(pred1)
	print(pred2)
	print(pred3)
	print(pred4)
	print(pred5)

if __name__ == '__main__':
	main()
