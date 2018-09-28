import pickle 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import numpy as np 
import gym 

from vae_model import VAE 




vae = VAE()
vae.load_state_dict(torch.load('trained_models/VAE_1'))
vae.eval()

def save_data(code, ac, done): 




class Env: 

    def __init__(self): 

        self.env = gym.make('CarRacing-v0')

    def reset(self): 

        return torch.tensor(self.env.reset()).float()

    def step(self, action): 

        ns, r, done, info = self.env.step(action)
        return torch.tensor(ns).float(), done

    def render(self): 
        self.env.render()

env = Env()
max_frames = 10000

path= '/home/mehdi/Codes/ML3/WorldModels/dataset_rnn/'

s = env.reset()

for frame in range(max_frames):

    action = np.random.uniform(-1.,1., (3))
    action[1] = np.random.uniform(0.3,1.)
    env.render()
    ns, done = env.step(action)
    
    code = vae.to_code(s).detach()
    save_data(code,action, done)

    if done: 
        s = env.reset()

    else: 
        s = ns










