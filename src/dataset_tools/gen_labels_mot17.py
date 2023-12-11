"""
将MOT17转换为yolo v5格式
obj_id, x0, y0, x1, y1, 0
"""

import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
import random

DATA_ROOT = '/data/wujiapeng/datasets/MOT17/'
VALID_CLASS = [0]

image_wh_dict = {}  # seq->(w,h) 字典 用于归一化

def generate_imgs_and_labels(opts):
    """
    产生图片路径的txt文件以及yolo格式真值文件
    """
    if opts.split == 'test':
        seq_list = os.listdir(osp.join(DATA_ROOT, 'test'))
    else:
        seq_list = os.listdir(osp.join(DATA_ROOT, 'train'))
        seq_list = [item for item in seq_list if 'SDP' in item]  # 只取一个FRCNN即可
        if 'val' in opts.split: opts.half = True  # 验证集取训练集的一半

    print('--------------------------')
    print(f'Total {len(seq_list)} seqs!!')
    print(seq_list)
    
    if opts.random: 
        random.shuffle(seq_list)

    # 定义类别 MOT只有一类
    CATEGOTY_ID = 0  # pedestrian

    # 定义帧数范围
    frame_range = {'start': 0.0, 'end': 1.0}
    if opts.half:  # half 截取一半
        frame_range['end'] = 0.5

    if opts.split == 'test':
        process_train_test(seqs=seq_list, frame_range=frame_range, cat_id=CATEGOTY_ID, split='test', norm_for_yolo=opts.norm)
    else:
        process_train_test(seqs=seq_list, frame_range=frame_range, cat_id=CATEGOTY_ID, split=opts.split, norm_for_yolo=opts.norm)
                

def process_train_test(seqs: list, frame_range: dict, cat_id: int = 0, split: str = 'trian', norm_for_yolo: bool = False) -> None:
    """
    处理MOT17的train 或 test
    由于操作相似 故另写函数

    """   

    # 记录当前seq的每个类别的id offset 即应该从哪个id开始
    start_id_in_seq = {cls_id: 0 for cls_id in VALID_CLASS}

    for seq in seqs:
        print(f'Dealing with {split} dataset...')

        img_dir = osp.join(DATA_ROOT, 'train', seq, 'img1') if split != 'test' else osp.join(DATA_ROOT, 'test', seq, 'img1') # 图片路径
        imgs = sorted(os.listdir(img_dir))  # 所有图片的相对路径
        seq_length = len(imgs)  # 序列长度

        ann_of_seq_path = os.path.join(img_dir, '../', 'gt', 'gt.txt') # GT文件路径
        ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容

        # 需要预处理出当前序列 每个类别的标注id: 训练需要的id的映射
        seq_id_map = {cls_id: set() for cls_id in VALID_CLASS}
        for cls_id in VALID_CLASS:
            seq_cls_anno = ann_of_seq[ann_of_seq[:, 6] == float(1), :]

            for row_idx in range(seq_cls_anno.shape[0]):
                seq_id_map[cls_id].add(seq_cls_anno[row_idx, 1])  # set 自动去重

        # 将seq_id_map的set改为list 方便后面索引
        seq_id_map_ = {cls_id: list() for cls_id in VALID_CLASS}
        for k, v in seq_id_map.items():
            seq_id_map_[k] = sorted(list(v))

        if split != 'test':           

            # 求解图片高宽
            img_eg = cv2.imread(osp.join(img_dir, imgs[0]))
            w0, h0 = img_eg.shape[1], img_eg.shape[0]  # 原始高宽

            ann_of_seq_path = os.path.join(img_dir, '../', 'gt', 'gt.txt') # GT文件路径
            ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容

            gt_to_path = osp.join(DATA_ROOT, 'labels_with_ids', split, seq)  # 要写入的真值文件夹
            # 如果不存在就创建
            if not osp.exists(gt_to_path):
                os.makedirs(gt_to_path)

            exist_gts = []  # 初始化该列表 每个元素对应该seq的frame中有无真值框
            # 如果没有 就在train.txt产生图片路径

            for idx, img in enumerate(imgs):
                # img 形如: img000001.jpg
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                # 第一步 产生图片软链接
                # print('step1, creating imgs symlink...')
                if opts.generate_imgs:
                    img_to_path = osp.join(DATA_ROOT, 'images', split, seq)  # 该序列图片存储位置

                    if not osp.exists(img_to_path):
                        os.makedirs(img_to_path)

                    os.symlink(osp.join(img_dir, img),
                                    osp.join(img_to_path, img))  # 创建软链接
                
                # 第二步 产生真值文件
                # print('step2, generating gt files...')
                ann_of_current_frame = ann_of_seq[ann_of_seq[:, 0] == float(idx + 1), :]  # 筛选真值文件里本帧的目标信息
                exist_gts.append(True if ann_of_current_frame.shape[0] != 0 else False)

                gt_to_file = osp.join(gt_to_path, img[: -4] + '.txt')

                with open(gt_to_file, 'w') as f_gt:
                    for i in range(ann_of_current_frame.shape[0]):    
                        if int(ann_of_current_frame[i][6]) == 1 and int(ann_of_current_frame[i][7]) == 1 \
                            and float(ann_of_current_frame[i][8]) > 0.25:
                            # bbox xywh 
                            obj_id = int(ann_of_current_frame[i][1])
                            x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                            x0, y0 = max(x0, 0), max(y0, 0)
                            w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])

                            x1, y1 = x0 + w, y0 + h 

                            x1, y1 = min(x1, w0), min(y1, h0)

                            track_id_ = seq_id_map_[cls_id].index(obj_id) + start_id_in_seq[cls_id] + 1

                            if not norm_for_yolo:
                                write_line = '{:d} {:d} {:d} {:d} {:d} {:d}\n'.format(
                                    obj_id, x0, y0, x1, y1, 0)
                            else:
                                xc, yc = x0 + w // 2, y0 + h // 2  # 中心点 x y
                                # 归一化
                                xc, yc = xc / w0, yc / h0
                                w, h = w / w0, h / h0

                                write_line = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                     track_id_, xc, yc, w, h)

                            f_gt.write(write_line)

                f_gt.close()

        else:  # test 只产生图片软链接
            for idx, img in enumerate(imgs):
                # img 形如: img000001.jpg
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                # 第一步 产生图片软链接
                # print('step1, creating imgs symlink...')
                if opts.generate_imgs:
                    img_to_path = osp.join(DATA_ROOT, 'images', split, seq)  # 该序列图片存储位置

                    if not osp.exists(img_to_path):
                        os.makedirs(img_to_path)

                    os.symlink(osp.join(img_dir, img),
                                osp.join(img_to_path, img))  # 创建软链接

        # 第三步 产生图片索引train.txt等
        print(f'generating img index file of {seq}')        
        to_file = os.path.join('./mot17/', split + '.txt')
        with open(to_file, 'a') as f:
            for idx, img in enumerate(imgs):
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                if split == 'test' or exist_gts[idx]:
                    f.write('MOT17/' + 'images/' + split + '/' \
                            + seq + '/' + img + '\n')

            f.close()

    

if __name__ == '__main__':
    if not osp.exists('./mot17'):
        os.system('mkdir mot17')

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train, test or val')
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--half', action='store_true', help='half frames')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of test dataset devide train dataset')
    parser.add_argument('--random', action='store_true', help='random split train and test')

    parser.add_argument('--norm', action='store_true', help='only true when used in yolo training')

    opts = parser.parse_args()

    generate_imgs_and_labels(opts)
    # python src/dataset_tools/gen_labels_mot17.py --split train --generate_imgs --norm