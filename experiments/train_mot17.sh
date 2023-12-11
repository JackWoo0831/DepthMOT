cd src
# CUDA_VISIBLE_DEVICES=1 python train.py mot --dataset uavdt --exp_id uavdt --load_model '../models/pretrained/coco_dla.pth' --gpus 0 --data_cfg '../src/lib/cfg/uavdt.json'
# CUDA_VISIBLE_DEVICES=0 python train.py mot --dataset uavdt --exp_id uavdt --arch hrnet_32 --gpus 0 --batch_size 8 --data_cfg '../src/lib/cfg/uavdt.json'
CUDA_VISIBLE_DEVICES=0 python train.py mot_depth --dataset mot17 --exp_id mot17 --arch dladepth_32 --batch_size 4 --num_epochs 15 --hide_data_time --load_model '../models/pretrained/coco_mono_dla_depth.pth' --gpus 0 --data_cfg '../src/lib/cfg/mot17.json'
cd ..
