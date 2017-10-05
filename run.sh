#!/bin/bash

# $1:   GPU index
# $2:   train or test

echo "THIS IS NEW"
CUDA_VISIBLE_DEVICES=$1 py_gxdai main.py  --phase $2 --dataset_name modelnet10 --logdir "./logs/localGlobal" --checkpoint_dir ./checkpoint/wgan_gp --batch_size 5
#CUDA_VISIBLE_DEVICES=$1 py_gxdai main.py  --phase $2 --dataset_name modelnet10 --logdir "./logs/contextEncode" --checkpoint_dir ./checkpoint/contextEncode --batch_size 5
