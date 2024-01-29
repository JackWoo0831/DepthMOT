cd src 
model_pth='../models/trained/wo_reid_w_depth_KITTI_20240127_epoch40.pth'
CUDA_VISIBLE_DEVICES=1 python track.py mot_depth --dataset kitti --arch dladepth_32 --batch_size 1 --test_kitti True --load_model $model_pth --conf_thres 0.05
# CUDA_VISIBLE_DEVICES=2 python track.py mot --dataset kitti --arch dla_34 --batch_size 1 --test_kitti True --load_model $model_pth --conf_thres 0.1
cd ../