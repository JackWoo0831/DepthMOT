cd src 
model_pth='../models/trained/wo_reid_w_depth_UAVDT_20231208_epoch15.pth'
CUDA_VISIBLE_DEVICES=0 python track.py mot_depth --dataset uavdt --arch dladepth_32 --batch_size 1 --test_uavdt True --load_model $model_pth --conf_thres 0.4
cd ..
