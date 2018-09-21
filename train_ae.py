import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from vae_model import VAE, AE

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

	path = '/home/mehdi/Codes/ML3/WorldModels/runs_ae/'
	try: 
		os.makedirs(path)
	except OSError: 
		pass 

	nb_files = len(glob.glob('{}*'.format(path)))
	writer = SummaryWriter('{}/Run {}'.format(path, nb_files))

	return writer 

def print_logs(epoch, recon_loss, digits = 6): 


	str_length = 45
	t0 = 'Epoch: '
	t1 = 'Reconstruction loss'
	# t2 = 'KL div'

	size = len(t1)
	size_full = size + digits

	# diff_2 = size - len(t2)+1
	# diff_3 = size - len(t3)+1

	s1 = '|' + '='*10 + ' {}{} '.format(t0, epoch)
	s2 = '| ' + t1 + ': {:.6f}'.format(recon_loss)
	# s3 = '| ' + t2 + ':' + ' '*diff_2 + '{:.6f}'.format(klloss) 
	# s4 = '| ' + t3 + ':' + ' '*diff_3 + '{:.6f}'.format(recon_loss + klloss) 

	spaces = [str_length - len(s) for s in[s1,s2]]
	# spaces = str_length - len(s1)

	counter = 0 
	for s, sp in zip([s1,s2], spaces): 
		inden = ' ' if counter > 0 else '='
		print(s + inden*(sp-1)+'|')
		counter += 1

	print('|' + '='*(str_length-2) + '|\n\n')


def image_to_writer(writer, step, x_real, x_recon): 

	for im, name in zip([x_real, x_recon], ['Real', 'Recon']): 

		assert im.shape[1] == 3, "Image channels dimension should be 3"

		grid = tutils.make_grid(im)
		writer.add_image(name, grid, step)

def run_sample(model, loader, args): 

	if args.use_cuda: 
		model.cpu()

	x = loader.sample(args.nb_samples)
	r = model(x)
	r = r.detach()
	

	if args.use_cuda: 
		model.cuda()

	return torch.cat([x,x,x],1), torch.cat([r,r,r],1)

def write_scalars(dico, writer, epoch): 

	for k in dico.keys(): 
		writer.add_scalar(k, dico[k], epoch)


def small_init(model): 

	for m in model._modules: 
		model._modules[m].weight.data.normal_(0.,0.02)
		model._modules[m].bias.data.zero_()

def train(model, loader, args, writer): 

	if args.use_cuda: 
		model.cuda()

	adam = optim.Adam(model.parameters(), lr = args.lr)

	for epoch in range(1, args.epochs+1): 

		epoch_recon_loss = 0.

		for nb_batch in range(args.num_batchs): 

			if args.use_cuda: 
				x = loader.sample_cuda(args.batch_size)
			else: 
				x = loader.sample(args.batch_size)


			recon_loss = model.compute_loss(x)

			adam.zero_grad()
			(recon_loss).backward()
			adam.step()

			epoch_recon_loss += recon_loss.item()

		epoch_recon_loss /= float(nb_batch)


		dico_loss = {'Losses/Reconstruction':epoch_recon_loss}
		write_scalars(dico_loss, writer, epoch)
		print_logs(epoch, epoch_recon_loss)

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
			current_sample, current_recon = run_sample(model, loader, args)
			image_to_writer(writer, epoch, current_sample, current_recon)



loader = Loader('/home/mehdi/Codes/ML3/WorldModels/dataset', 10000)

writer = initialize_writer()
args = get_args()

ae = AE()
if args.load_model: 
	ae.load_state_dict(torch.load('/home/mehdi/Codes/ML3/WorldModels/trained_models/{}'.format(args.run_id)))
	print('Model loaded')
else:
	small_init(ae)


train(ae, loader, args, writer)

