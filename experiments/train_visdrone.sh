cd src
# CUDA_VISIBLE_DEVICES=0 python train.py mot --dataset visdrone --exp_id visdrone --load_model '../models/pretrained/coco_dla.pth' --gpus 0 --data_cfg '../src/lib/cfg/visdrone.json'
# CUDA_VISIBLE_DEVICES=0 python train.py mot --dataset visdrone --exp_id visdrone --arch hrnet_32 --gpus 0 --batch_size 8 --data_cfg '../src/lib/cfg/visdrone.json'
CUDA_VISIBLE_DEVICES=0 python train.py mot_depth --dataset visdrone --exp_id visdrone --arch dladepth_32 --batch_size 4 --num_epochs 10 --hide_data_time --load_model '../models/pretrained/coco_mono_dla_depth.pth' --gpus 0 --data_cfg '../src/lib/cfg/visdrone.json'
cd ..
