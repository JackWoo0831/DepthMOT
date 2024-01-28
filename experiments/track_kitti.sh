cd src 
model_pth='../exp/mot_depth/kitti/model_40.pth'
CUDA_VISIBLE_DEVICES=1 python track.py mot_depth --dataset kitti --arch dladepth_32 --batch_size 1 --test_kitti True --load_model $model_pth --conf_thres 0.05
cd ..
