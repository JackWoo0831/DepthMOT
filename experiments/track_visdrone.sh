cd src 
model_pth='../models/trained/wo_reid_w_depth_20231026_epoch10_1088x608.pth'
CUDA_VISIBLE_DEVICES=1 python track.py mot_depth --dataset visdrone --arch dladepth_32 --batch_size 1 --test_visdrone True --load_model $model_pth --conf_thres 0.4
cd ..
