"""
merge dla ckpts and monodepth ckpts
"""

import torch
import os 
import os.path as osp

# depth decoder model attr -> name in state dict 
DEPTH_DECODER_LIST = [
    'upconv_4_0', 
    'upconv_4_1', 
    'upconv_3_0', 
    'upconv_3_1', 
    'upconv_2_0', 
    'upconv_2_1', 
    'upconv_1_0', 
    'upconv_1_1', 
    'upconv_0_0', 
    'upconv_0_1', 
    'dispconv_0', 
    'dispconv_1', 
    'dispconv_2', 
    'dispconv_3', 
]

DEPTH_DECODER_DICT = {k: str(v) for v, k in enumerate(DEPTH_DECODER_LIST)}
DEPTH_DECODER_INVERSE_DICT = {str(v): k for v, k in enumerate(DEPTH_DECODER_LIST)}

POSE_DEOCDER_LIST = [
    'squeeze', 
    'pose_0', 
    'pose_1', 
    'pose_2', 
]

POSE_DECODER_DICT = {k: str(v) for v, k in enumerate(POSE_DEOCDER_LIST)}
POSE_DECODER_INVERSE_DICT = {str(v): k for v, k in enumerate(POSE_DEOCDER_LIST)}



if __name__ == '__main__':
    dla_model_path = 'models/pretrained/coco_dla.pth'
    mono_model_path = 'models/pretrained/mono_1024x320'
    save_path = 'models/pretrained/coco_mono_dla_depth.pth'

    save_ckpt = {}

    # copy whole dla state dict into save_ckpt
    dla_model = torch.load(dla_model_path)
    dla_model = dla_model['state_dict']

    for k in dla_model.keys():
        if k.startswith('module') and not k.startswith('module_list'):
            save_ckpt[k[7: ]] = dla_model[k]  # convert data_parallal to model

        else:
            save_ckpt[k] = dla_model[k]

    # copy depth decoder state dict into save_ckpt
    depth_decoder_model = torch.load(osp.join(mono_model_path, 'depth.pth'))

    for k in depth_decoder_model.keys():
        # model: depth_decoder.upconv_4_0.... -> decoder.0....
        layer_idx = k.split('.')[1]
        save_key_name = 'depth_decoder.' + DEPTH_DECODER_INVERSE_DICT[layer_idx] + \
                        k[7 + 1 + len(layer_idx): ]
        
        save_ckpt[save_key_name] = depth_decoder_model[k]

    # copy pose encoder (separate mode)
    pose_encoder_model = torch.load(osp.join(mono_model_path, 'pose_encoder.pth'))
    for k in pose_encoder_model:
        # pose_encoder.encoder... -> encoder...
        save_key_name = 'pose_encoder.' + k 
        save_ckpt[save_key_name] = pose_encoder_model[k]

    # copy pose decoder 

    pose_decoder_model = torch.load(osp.join(mono_model_path, 'pose.pth'))

    for k in pose_decoder_model:
        # pose_decoder.squeeze.... -> net.0...
        layer_idx = k.split('.')[1]
        save_key_name = 'pose_decoder.' + POSE_DECODER_INVERSE_DICT[layer_idx] + \
                        k[3 + 1 + len(layer_idx): ]
        
        save_ckpt[save_key_name] = pose_decoder_model[k]

    save_ckpt_ = {}
    save_ckpt_['epoch'] = 30 
    save_ckpt_['state_dict'] = save_ckpt

    torch.save(save_ckpt_, save_path)
    

    # python src/merge_ckpt.py 