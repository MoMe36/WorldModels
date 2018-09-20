import gym 
import numpy as np 
import pickle 
import os 

import matplotlib.pyplot as plt 

from PIL import Image 

env = gym.make('CarRacing-v0')

# input(env.action_space.)
action_shape = env.action_space.shape[0] 
image_size = [64,64]


max_frames = 10000
path = '/home/mehdi/Codes/ML3/WorldModels/dataset'




s = env.reset()
for frame in range(max_frames): 

	if frame%10 == 0: 
		action = np.random.uniform(-1.,1., (action_shape))
		action[1] = np.random.uniform(0.2,1.)

	s, r, done, _ = env.step(action)
	env.render()
	if done: 
		env.reset()

	# img = np.clip(s.astype(float)/255., 0., 1.)
	im = Image.fromarray(s)
	im = im.resize(image_size)

	im = np.array(im)
	im = np.transpose(im, [2,0,1])
	im = np.clip(im.astype(float)/255., 0.,1.)

	im = np.expand_dims(np.mean(im, 0), axis = 0)

	# if frame > 0 and frame % 100 ==0: 
	# 	plt.matshow(im[0], cmap = 'gray')
	# 	plt.pause(0.1)
	# 	input()
		
	pickle.dump(im, open(os.path.join(path,str(frame)), 'wb'))

	if(frame%1000) == 0:
		print('Frame {}/{}'.format(frame, max_frames))







