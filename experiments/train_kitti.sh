cd src
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py mot_depth --dataset kitti --exp_id kitti --arch dladepth_32 --batch_size 8 --num_epochs 20 --hide_data_time --load_model '../exp/mot_depth/kitti/model_20.pth' --gpus 0 --data_cfg '../src/lib/cfg/kitti.json'
cd ..
