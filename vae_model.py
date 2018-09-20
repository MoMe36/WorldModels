import torch 
import torch.nn as nn 
import torch.nn.functional as F 



class VAE(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)


		self.c1 = nn.Conv2d(1,32,4, stride = 2)
		self.c2 = nn.Conv2d(32,64,4, stride = 2)
		self.c3 = nn.Conv2d(64,128,4, stride = 2)
		self.c4 = nn.Conv2d(128,256,4, stride = 2)

		self.mean = nn.Linear(1024,32) 
		self.logvar = nn.Linear(1024,32)

		self.z_to_vec = nn.Linear(32, 1024)

		self.d1 = nn.ConvTranspose2d(1024,128,5, stride = 2)
		self.d2 = nn.ConvTranspose2d(128,64,5, stride = 2)
		self.d3 = nn.ConvTranspose2d(64,32,6, stride = 2)
		self.d4 = nn.ConvTranspose2d(32,1,6, stride = 2)

	def encode(self, x): 

		batch_size = x.shape[0]
		conv_layers = [self.c1, self.c2, self.c3, self.c4]

		for l in conv_layers: 
			x = F.relu(l(x))

		x = x.reshape(batch_size, -1)

		means = self.mean(x)
		logvar = self.logvar(x)

		return means, logvar

	def sample(self, batch_size): 

		x = torch.randn(batch_size, 32)
		out = self.decode(x)

		return out.detach()

	def forward(self, x): 

		means, logvars = self.encode(x)
		z = self.reparametrize(means, logvars)
		recon = self.decode(z)

		return recon, z, means, logvars
		
	def reparametrize(self, mean, logvar): 

		if self.training: 
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			sample = mean + eps*std

			return sample 
		else: 
			return mean 

	def decode(self, x): 

		batch_size = x.shape[0]

		x = F.relu(self.z_to_vec(x))

		deconv_layers = [self.d1, self.d2, self.d3]
		x = x.reshape(batch_size, -1,1,1)
		for l in deconv_layers: 
			x = F.relu(l(x))

		return torch.sigmoid(self.d4(x))

	def compute_loss(self, batch): 

		recon, z, means, logvars = self(batch)

		recon_loss = F.binary_cross_entropy(recon, batch, reduction = 'sum') 
		kl_d_loss = -0.5*torch.sum(1 + logvars - means.pow(2) - logvars.exp())

		return recon_loss, kl_d_loss



