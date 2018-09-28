import torch 
import numpy as np 
import pickle
from argparse import ArgumentParser 
import os 
import random


def get_args():


	parser = ArgumentParser()

	parser.add_argument('--nb_samples', type = int, default = 16)
	parser.add_argument('--lr', type = float, default = 3e-4)
	parser.add_argument('--epochs', type = int, default = 5000)
	parser.add_argument('--num_batchs', type = int, default = 16)
	parser.add_argument('--batch_size', type = int, default = 32)
	parser.add_argument('--save_interval', type = int, default = 10)
	parser.add_argument('--im_interval', type = int, default = 250)
	parser.add_argument('--run_id', default = "v1")
	parser.add_argument('--use_cuda', default = "n")
	parser.add_argument('--load_model', default = "n")



	args = parser.parse_args()
	args.use_cuda = True if args.use_cuda in ['y', 'Y', 'o', 'O'] else False 
	args.load_model = True if args.load_model in ['y', 'Y', 'o', 'O'] else False 

	return args


def load(name): 

	x = pickle.load(open(name, 'rb'))
	return x 


class Loader: 

	def __init__(self, path, max_im): 

		self.path = path 
		self.max_im = max_im 

	def sample(self, batch_size = 64): 

		batch = np.zeros((batch_size, 1, 64,64))
		inds = random.sample(list(np.arange(self.max_im)), batch_size)
		for i, ind in enumerate(inds):

			batch[i,:,:,:] = load(os.path.join(self.path, str(ind)))

		return torch.tensor(batch).float()

	def sample_cuda(self, batch_size = 64): 

		batch = self.sample(batch_size)
		return batch.cuda()