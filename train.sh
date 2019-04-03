#!/usr/bin/env bash
source ~/minming/torch/bin/activate

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                    --dataset kiktech_20181001 --net res101 \
#                    --bs 6 --nw 2 \
#                    --lr 0.01 --lr_decay_step 5 \
#                    --cuda  --epochs 20 


#CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                   --dataset kiktech_20181001 --net mobilenet \
#                   --bs 6 --nw 2 \
#                   --lr 0.01 --lr_decay_step 5 \
#                   --cuda  --epochs 20

#CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                   --dataset kiktech_2018joint10 --net mobilenet \
#                   --bs 6 --nw 2 \
#                   --lr 0.1 --lr_decay_step 5 \
#                   --cuda  --epochs 50

#CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                   --dataset kiktech_2018joint10 --net mobilenet \
#                   --bs 6 --nw 2 \
#                   --lr 0.01 --lr_decay_step 5 \
#                   --cuda  --epochs 500 \
#                   --start_epoch 156 \
#                   --r True --checksession 1 --checkepoch 156 --checkpoint 1


#CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                   --dataset kiktech_2018joint-480p-147 --net mobilenet \
#                   --bs 6 --nw 2 \
#                   --lr 0.1 --lr_decay_step 5 \
#                   --cuda  --epochs 50


CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset kiktech_2018joint-480p-147 --net shufflenet \
                   --bs 1 --nw 2 \
                   --lr 0.1 --lr_decay_step 5 \
                   --cuda  --epochs 50

deactivate
