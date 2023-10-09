#!/usr/bin/env bash

#GPUs
gpus=0

#Set paths
checkpoint_root=/home/amarachi/CheXRelFormer/checkpoints/checkpoints_final



data_name=CXRData


img_size=256
batch_size=4  
lr=0.00006        
max_epochs=100
embed_dim=256

net_G=ConvolutionalDifferenceCNN   #change this to the model you want to use

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                     
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights

#pretrain=//home/amarachi/CheXRelFormer/checkpoints/ChangeFormer_DSIFN/best_ckpt.pt

#Train and Validation splits
split=train         #trainval
split_val=val      #test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

CUDA_VISIBLE_DEVICES=0 python /home/amarachi/CheXRelFormer/main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root}  --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}