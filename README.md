# World Models VAE 


This repo holds an implementation of the VAE model detailed in [WorldModels](https://github.com/worldmodels.gituhub.io)

## Results



## Insights 

While implementing, I came across some facts that I believe are worth sharing: 

* The mean square error loss must avoid averaging across the batch. Make sure to use `F.mse_loss(reconstruction, real_sample, reduction = 'sum')
* I had better results dividing the KL-divergence loss by the batch size: `kl_loss/float()batch_size)

 
