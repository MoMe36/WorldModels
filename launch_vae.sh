#!/bin/bash


python train_vae.py --run_id "VAE_1" --use_cuda y --im_interval 100
# python train_ae.py --run_id "AE_test" --use_cuda y --im_interval 200 --load_model y