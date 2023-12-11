import math
import os
import os.path as osp
import random
from collections import OrderedDict

import cv2
import numpy as np
import copy

from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh

from .jde import JointDataset, letterbox 

from PIL import Image
from torchvision.transforms import Resize, ColorJitter

class UAVDataset_Depth(JointDataset):
    """
    DataLoader with depth support
    """
    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None, 
                 frame_interval=5, ):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 2 if opt.dataset == 'visdrone' else 1

        self.frame_interval = frame_interval
        self.img_name_prefix_length = 3 if opt.dataset == 'uavdt' else 0


        # read image paths and get gt file paths
        # when reading, delete the first and last $frame_interval$ frames of each seq for depth estimation
        for ds, path in paths.items():
            with open(path, 'r') as file:
                img_paths = file.readlines()
                cur_dataset_imgs = []

                for row_idx, row in enumerate(img_paths):
                    row = row.strip()
                    if not len(row): continue

                    # if first $frame_interval$ row of first seq, skip
                    if row_idx < frame_interval: continue

                    # if first $frame_interval$ row of second, thrid, ..., seq, delete the last 
                    # $frame_interval$ frames of previous seq
                    frame_idx = row.split('/')[-1]
                    frame_idx = int(frame_idx[self.img_name_prefix_length: -4])
                    
                    if frame_idx == 1:
                        cur_dataset_imgs = cur_dataset_imgs[:-frame_interval]
                        continue
                    elif frame_idx < frame_interval + 1:
                        continue

                    cur_dataset_imgs.append(os.path.join(root, row))
                
                # delete the last $frame_interval$ frames of last seq
                cur_dataset_imgs = cur_dataset_imgs[:-frame_interval]

                self.img_files[ds] = cur_dataset_imgs

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]
            
        
            
        # get max_index: max object id nunmber in whole dataset, 
        # self.tid_num: dataset_name: max_index
        for ds, label_paths in self.label_files.items():

            cls_max_id = {cls_id: 0 for cls_id in range(self.num_classes)}  # dict that map cls_id to max track_id
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue

                lb = lb.reshape((-1, 6))  # in case lb is only one row

                for row in lb:
                    # maintain the max ID corresponding to the cls
                    if row[1] > cls_max_id[int(row[0])]:
                        cls_max_id[int(row[0])] = int(row[1])

            self.tid_num[ds] = cls_max_id

        # obj id in every dataset begin with 0
        # now we need to add offset between different datasets
        self.start_id_in_all = {ds: dict() for ds in dataset_names}
        cls_last_id = {cls_id: 0 for cls_id in range(self.num_classes)}  # cls -> id, mark among all datasets, the last id of the cls

        for i, (ds, cls_max_id) in enumerate(self.tid_num.items()):
            for cls_id, max_id in cls_max_id.items():
                self.start_id_in_all[ds][cls_id] = cls_last_id[cls_id]
                cls_last_id[cls_id] += max_id

        self.nID = cls_last_id  # cls -> id among all datasets

        self.nds = [len(x) for x in self.img_files.values()]  # image numbers
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]  # like a accumulate of number of images
        self.nF = sum(self.nds)  # number of all frames


        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        self.resize = Resize(size=(self.height, self.width), interpolation=Image.ANTIALIAS)
        self.color_augment = ColorJitter.get_params(brightness=(0.8, 1.2), contrast=(0.8, 1.2), 
                                                    saturation=(0.8, 1.2), hue=(-0.1, 0.1))

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path)

        # get prev and next frame img for depth 
        # NOTE: check img preprocess in monodepth2

        imgs_prev = self.get_img_only(self._gen_img_path(img_path, -self.frame_interval))
        imgs_next = self.get_img_only(self._gen_img_path(img_path,  self.frame_interval))

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                # map track id in inner dataset to all dataset
                cls_id = int(labels[i, 0])
                labels[i, 1] += self.start_id_in_all[ds][cls_id]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        if self.opt.reid:
            track_id_map = np.zeros((num_classes, output_h, output_w), dtype=int)
            cls_id_map = -1 * np.ones((1, output_h, output_w), dtype=int)
        else:
            track_id_map, cls_id_map = None, None

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(min(num_objs, self.max_objs)):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1] - 1
                bbox_xys[k] = bbox_xy

                if self.opt.reid:
                    cls_id_map[0, ct_int[1], ct_int[0]] = cls_id 
                    track_id_map[cls_id, ct_int[1], ct_int[0]] = label[1] - 1

        if cls_id_map is not None:
            ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys,
                   'cls_id_map': cls_id_map, 'track_id_map': track_id_map, 
                   'prev_input': imgs_prev, 'next_input': imgs_next}
        else:
            ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'bbox': bbox_xys, 
                   'prev_input': imgs_prev, 'next_input': imgs_next}
        return ret
    
    def get_data(self, img_path, label_path):

        img = self._pil_loader(img_path)  # RGB

        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        
        img_resized = self.resize(img)

        w0, h0 = img.size

        if self.augment:
            img_resized = self.color_augment(img_resized)

        # process labels
        # scale_factor_h, scale_factor_w = self.height / h0, self.width / w0

        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0

        else:
            labels = np.array([])       

        if self.transforms is not None:
            img_resized = self.transforms(img_resized)  # normed and to tensor

        return img_resized, labels, img_path, (h0, w0)
    
    def get_img_only(self, img_path):
        """
        Only return the resized img (class: torch.Tensor)

        partly copied from self.get_data()
        """

        img = self._pil_loader(img_path)  # RGB

        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        
        img_resized = self.resize(img)

        if self.transforms is not None:
            img_resized = self.transforms(img_resized)

        return img_resized
    
    def _gen_img_path(self, img_path, frame_bias):
        """
        gen prev and next img path for depth estimation
        frame_bias: +frame_interval or -frame_interval
        """
        frame_name = img_path.split('/')[-1]
        tail_length = len(frame_name)

        new_frame_idx = int(frame_name[self.img_name_prefix_length: -4]) + frame_bias

        prefix = frame_name[:self.img_name_prefix_length]

        num_of_digits = tail_length - self.img_name_prefix_length - 4

        new_frame_name = prefix + '{:0{nd}d}.jpg'.format(new_frame_idx, nd=num_of_digits)

        return img_path[:-tail_length] + new_frame_name
    
    def _pil_loader(self, img_path):
        """
        use PIL to load image
        """
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')