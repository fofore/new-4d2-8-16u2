#!/usr/bin/env bash
source /home/jie/pyvienv/pytorch0.4.0/bin/activate

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
#                    --dataset kiktech_20181001 --net res101 \
#                    --bs 6 --nw 2 \
#                    --lr 0.01 --lr_decay_step 5 \
#                    --cuda  --epochs 20 \

# python demo_kiktech.py --dataset kiktech_20181001 --image_dir images-kik \
#        --net res101  --checksession 1 --checkepoch 20 --checkpoint 218  --cuda --load_dir ./models

#python demo_kiktech.py --dataset kiktech_20181001 --image_dir images-kik \
#       --net mobilenet  --checksession 1 --checkepoch 20 --checkpoint 218  --cuda --load_dir ./models

#python demo_kiktech.py --dataset kiktech_2018joint10 --image_dir images-kik-joint \
#       --net mobilenet  --checksession 1 --checkepoch 50 --checkpoint 1  --cuda --load_dir ./models

#python demo_kiktech.py --dataset kiktech_2018joint-480p-147 --image_dir images-kik-jointnew \
#       --net mobilenet  --checksession 1 --checkepoch 44 --checkpoint 36  --cuda --load_dir ./models \
#       --data_path ./data/kiktech/kiktech2018joint-480p-147


python demo_kiktech.py --dataset kiktech_2018joint-480p-147 --image_dir images-kik-jointnew \
       --net shufflenet  --checksession 1 --checkepoch 249 --checkpoint 219  --cuda --load_dir ./models \
       --data_path ./data/kiktech/kiktech2018joint-480p-147

deactivate
