# World Models VAE 


This repo holds an implementation of the VAE model detailed in [WorldModels](https://github.com/worldmodels.gituhub.io)

## Results
You can see the results for yourself using the runs folder. Just call Tensorboard to do so. 
However, here are some images: 

[Sampling 2500](imgs/recon25.png)
[Sampling 4900](imgs/recon_49.png)

## Insights 

While implementing, I came across some facts that I believe are worth sharing: 

* The mean square error loss must avoid averaging across the batch. Make sure to use `F.mse_loss(reconstruction, real_sample, reduction = 'sum')
* I had better results dividing the KL-divergence loss by the batch size: `kl_loss/float()batch_size)

 
