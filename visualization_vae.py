import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider, Button 

from arguments import Loader, get_args 
from vae_model import VAE 


args = get_args()



loader = Loader('./dataset', 10000)
vae = VAE()
vae.load_state_dict(torch.load('./trained_models/{}'.format(args.run_id)))


vae.eval() 


# Assuming code dimension = 32
nb_axes = 32 
ax_margin = 0.02
ax_width = 0.1
ax_updown_margin = 0.1 
ax_left = 0.85
inc = (1. - 2.*ax_updown_margin)/(float(nb_axes))


f, ax = plt.subplots()
plt.subplots_adjust(left = 0.1, right= 0.8)

slider_min = -2.
slider_max = 2.

all_axes = [plt.axes([ax_left,ax_updown_margin + i*inc,ax_width, ax_margin]) for i in range(nb_axes)]
all_sliders = [Slider(axe, '', slider_min, slider_max) for i,axe in enumerate(all_axes)]

reset_ax = plt.axes([0.05,0.1,0.1,0.05])
reset_button = Button(reset_ax, 'Reset', hovercolor = '0.975')

def init_drawing(): 
	im, recon, z = reset_viz()
	full_image = set_full_image(im, recon)

	for zi, s in zip(z, all_sliders): 
		s.val = zi

	ax.clear()
	m = ax.matshow(full_image, cmap = 'gray')
	return m 

def set_full_image(image, recon, margin = 10): 

	full_image = np.ones((64,64*2+margin))
	full_image[:,0:64] = image
	full_image[:,64+margin:] = recon

	return full_image

def reset_viz(): 

	image = loader.sample(1)
	recon, z, _ , _ = vae(image)

	image = image.numpy().reshape(64,64)
	recon = recon.detach().numpy().reshape(64,64)
	z = z.detach().numpy().reshape(-1)

	return image, recon, z  

def reset_and_set(event): 
	im, recon, z = reset_viz()
	full_image = set_full_image(im, recon)

	for zi, s in zip(z, all_sliders): 
		s.val = zi

	ax.clear()
	m = ax.matshow(full_image, cmap = 'gray')


def update_vector(val): 

	values = np.zeros((len(all_sliders)))
	for i, slider in enumerate(all_sliders): 
		values[i] = slider.val

	tensor = torch.tensor(values).float().reshape(1,-1)
	output = vae.decode(tensor).detach().numpy().reshape(64,64)

	current_full_image = ax.images[0]._A # this get the matrix displayed
	current_full_image[:,74:] = output

	# full_image = set_full_image(image, output)
	ax.clear()
	ax.matshow(current_full_image, cmap = 'gray')

init_drawing()


for slider in all_sliders: 
	slider.on_changed(update_vector)

reset_button.on_clicked(reset_and_set)

plt.show()
