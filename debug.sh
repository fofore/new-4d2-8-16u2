rm /home/anton/minming/kiktech/dection-segmentation/data/cache/kiktech_2018joint-480p-147_trainval_gt_roidb.pkl

CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                   --dataset kiktech_2018joint-480p-147 --net shufflenet \
                   --bs 1 --nw 2 \
                   --lr 0.1 --lr_decay_step 5 \
                   --cuda  --epochs 40 --use_tfb
