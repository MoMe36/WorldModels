import torch
import torch.nn as nn 
import torch.nn.functional as F 
from vae_model import VAE 

from arguments import get_args 

from tensorboardX import SummaryWriter 
from torchvision import utils as tutils

import numpy as np 
import random 
import pickle 

import os 
import glob 


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

def initialize_writer(): 

	path = '/home/mehdi/Codes/ML3/WorldModels/runs/'
	try: 
		os.makedirs(path)
	except OSError: 
		pass 

	nb_files = len(glob.glob('*'))
	writer = SummaryWriter('Run {}'.format(nb_files))

	return writer 

def image_to_writer(writer, step, x_real, x_recon, x_sampled): 

	for im, name in zip([x_real, x_recon, x_sampled], ['Real', 'Recon', 'Samples']): 

		assert im.shape[1] == 3, "Image channels dimension should be 3"

		grid = tutils.make_grid(im)
		writer.add_image(name, grid, step)

def run_sample(model, loader, args): 

	if args.use_cuda: 
		model.cpu()

	x = loader.sample(args.nb_samples)
	r,_,_,_ = model(x).detach()
	z = model.sample(args.nb_samples).detach()


	if args.use_cuda: 
		model.cuda()

	return torch.cat([x,x,x],1), torch.cat([r,r,r],1), torch.cat([z,z,z],1), 

def train(model, loader, args, writer): 

	if args.use_cuda: 
		model.cuda()

	adam = optim.Adam(model.parameters(), lr = args.lr)

	for epoch in range(args.epochs): 

		epoch_recon_loss = 0.
		epoch_kl_d_loss = 0.

		for nb_batch in range(args.num_batchs): 

			if args.use_cuda: 
				x = loader.sample_cuda(args.batch_size)
			else: 
				x = loader.sample(args.batch_size)


			recon_loss, kl_d_loss = model.compute_loss(x)

			adam.zero_grad()
			(recon_loss + kl_d_loss).backward()
			adam.step()

			epoch_recon_loss += recon_loss.item()
			epoch_kl_d_loss += kl_d_loss.item()

		epoch_recon_loss /= float(nb_batch)
		epoch_kl_d_loss /= float(nb_batch)

		writer.add_scalars('Losses', {'Reconstruction':epoch_recon_loss , 'KL_Div':epoch_kl_d_loss , 'Full':epoch_recon_loss + epoch_kl_d_loss}, epoch)

		if epoch%args.save_interval == 0: 
			save_path = '/home/mehdi/Codes/ML3/WorldModels/trained_models/' 
			try: 
				os.makedirs(save_path)
			except OSError: 
				pass 

			if args.use_cuda: 
				model.cpu() 
			torch.save(model.state_dict(),save_path + args.run_id)
			if args.use_cuda: 
				model.cuda()

		if(epoch%args.im_interval == 0): 
			current_sample, current_recon, current_im = run_sample(model, loader)
			image_to_writer(writer, epoch, current_sample, current_recon, current_im)



loader = Loader('/home/mehdi/Codes/ML3/WorldModels/dataset', 10000)

writer = initialize_writer()