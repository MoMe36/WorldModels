import numpy as np 
import matplotlib.pyplot as plt 
import os 

import pickle 
def load(name): 

	x = pickle.load(open(name, 'rb'))
	return x 

class Loader: 

	def __init__(self, path, max_im): 

		self.path = path 
		self.max_im = max_im 

	def sample(self): 

		ind = np.random.randint(0, self.max_im)

		x = load(os.path.join(self.path, str(ind)))

		return x.reshape(64,64), ind 

loader = Loader('/home/mehdi/Codes/ML3/WorldModels/dataset', 10000)

f, ax = plt.subplots()
for i in range(1000): 
	x, nb = loader.sample()

	
	ax.matshow(x, cmap = 'gray')
	ax.set_title('Image {}'.format(nb))
	plt.pause(0.1)
	input()

	ax.clear()

plt.show()