cd src
# CUDA_VISIBLE_DEVICES=2 python -W ignore train.py mot_depth --dataset kitti --exp_id kitti --arch dladepth_32 --batch_size 12 --num_epochs 20 --hide_data_time --load_model '../models/pretrained/coco_mono_dla_depth.pth' --gpus 0 --data_cfg '../src/lib/cfg/kitti.json'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py mot_depth --dataset kitti --exp_id kitti --arch dladepth_32 --batch_size 12 --num_epochs 40 --hide_data_time --resume --lr_step 45 --lr 0.00001 --gpus 0 --data_cfg '../src/lib/cfg/kitti.json'
cd ..
