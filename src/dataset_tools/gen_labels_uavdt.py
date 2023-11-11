"""
将UAVDT转换为fairmot格式
cls_id, obj_id, xc, yc, w, h
"""

import os
import os.path as osp
import argparse
import cv2
import glob
import numpy as np
import random

DATA_ROOT = '/data/wujiapeng/datasets/UAVDT/'
TEST_SEQ = ['M1008', 'M1003', 'M1002', 'M0203', 'M0402', 'M0202', 'M0604', 'M0601', 'M0101', 'M1202']

VALID_CLASS = [0]

image_wh_dict = {}  # seq->(w,h) 字典 用于归一化

def generate_imgs_and_labels(opts):
    """
    产生图片路径的txt文件以及yolo格式真值文件
    """
    seq_list = os.listdir(osp.join(DATA_ROOT, 'UAV-benchmark-M'))
    print('--------------------------')
    print(f'Total {len(seq_list)} seqs!!')
    # 划分train test
    if opts.random: 
        random.shuffle(seq_list)

        bound = int(opts.ratio * len(seq_list))
        train_seq_list = seq_list[: bound]
        test_seq_list = seq_list[bound:]
        del bound

    else:
        train_seq_list = [seq for seq in seq_list if seq not in TEST_SEQ]
        test_seq_list = [seq for seq in seq_list if seq in TEST_SEQ]

    print(f'train dataset: {train_seq_list}')
    print(f'test dataset: {test_seq_list}')
    print('--------------------------')
    
    if not osp.exists('./uavdt/'):
        os.makedirs('./uavdt/')

    # 定义类别 UAVDT只有一类
    CATEGOTY_ID = 0  # car

    # 定义帧数范围
    frame_range = {'start': 0.0, 'end': 1.0}
    if opts.half:  # half 截取一半
        frame_range['end'] = 0.5

    # 分别处理train与test
    # process_train_test(train_seq_list, frame_range, CATEGOTY_ID, 'train', norm_for_yolo=opts.norm, debug=opts.debug)
    process_train_test(test_seq_list, {'start': 0.0, 'end': 1.0}, CATEGOTY_ID, 'test', norm_for_yolo=opts.norm, debug=opts.debug)
    print('All Done!!')
                

def process_train_test(seqs: list, frame_range: dict, cat_id: int = 0, split: str = 'trian', norm_for_yolo: bool = False,
                       debug: bool = False) -> None:
    """
    处理UAVDT的train 或 test
    由于操作相似 故另写函数

    """   

    # 记录当前seq的每个类别的id offset 即应该从哪个id开始
    start_id_in_seq = {cls_id: 0 for cls_id in VALID_CLASS}

    for seq in seqs:
        print(f'Processing seq {seq}')

        img_dir = osp.join(DATA_ROOT, 'UAV-benchmark-M', seq, 'img1')  # 图片路径
        imgs = sorted(os.listdir(img_dir))  # 所有图片的相对路径
        seq_length = len(imgs)  # 序列长度

        # 求解图片高宽
        img_eg = cv2.imread(osp.join(img_dir, imgs[0]))
        w0, h0 = img_eg.shape[1], img_eg.shape[0]  # 原始高宽

        ann_of_seq_path = os.path.join(img_dir, '../', 'gt', 'gt.txt') # GT文件路径
        ann_of_seq = np.loadtxt(ann_of_seq_path, dtype=np.float32, delimiter=',')  # GT内容

        exist_gts = []  # 初始化该列表 每个元素对应该seq的frame中有无真值框
        # 如果没有 就在train.txt产生图片路径

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

        # 储存每个序列中每个类别的最大ID号是多少
        # 这样在进行下一个序列遍历的时候 ID在此基础上累加
        max_id_in_seq = {cls_id: 0 for cls_id in VALID_CLASS}
        for key in seq_id_map.keys():
            max_id_in_seq[key] = len(seq_id_map[key])

        # 打印当前seq的id信息
        print(f'===In seq {seq}, id info:')
        for cls_id in VALID_CLASS:
            print(f'===cls: {cls_id}, start_id: {start_id_in_seq[cls_id]}, max_id: {max_id_in_seq[cls_id]}')

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

            write_lines = []
            for i in range(ann_of_current_frame.shape[0]):    
                if int(ann_of_current_frame[i][6]) == 1:
                    # bbox xywh 
                    track_id = int(ann_of_current_frame[i][1])
                    x0, y0 = int(ann_of_current_frame[i][2]), int(ann_of_current_frame[i][3])
                    w, h = int(ann_of_current_frame[i][4]), int(ann_of_current_frame[i][5])
                    
                    xc, yc = x0 + w * 0.5, y0 + h * 0.5

                    xc_norm, yc_norm = xc / w0, yc / h0
                    w_norm, h_norm = w / w0, h / h0

                    track_id_ = seq_id_map_[cls_id].index(track_id) + start_id_in_seq[cls_id] + 1

                    write_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        0, track_id_, xc_norm, yc_norm, w_norm, h_norm)
                    
                    write_lines.append(write_str)

            to_file = osp.join(DATA_ROOT, 'labels_with_ids', split, seq)
            if not osp.exists(to_file):
                os.makedirs(to_file)

            to_file = osp.join(to_file, 'img' + str(idx + 1).zfill(6) + '.txt')

            with open(to_file, 'a') as f_to:
                for line in write_lines:
                    f_to.write(line)

            # debug 标签可视化
            if debug and idx < 30:  # 每个序列可视化前30帧
                vis_labels(img_path=osp.join(DATA_ROOT, 'UAV-benchmark-M', seq, 'img1', img), annos=write_lines)


        # 第三步 产生图片索引train.txt等
        print(f'generating img index file of {seq}')        
        to_file = os.path.join('./src/data/uavdt.txt')
        with open(to_file, 'a') as f:
            for idx, img in enumerate(imgs):
                if idx < int(seq_length * frame_range['start']) or idx > int(seq_length * frame_range['end']):
                    continue
                
                if exist_gts[idx]:
                    # f.write('UAVDT/' + 'images/' + split + '/' \
                    #         + seq + '/' + img + '\n')
                    
                    f.write(osp.join(DATA_ROOT, 'images', split, seq, img) + '\n')

            f.close()

        # 更新id offset
        for cls_id in VALID_CLASS:
            start_id_in_seq[cls_id] += max_id_in_seq[cls_id]

def vis_labels(img_path, annos, write_path='./debug_datasets/'):
    """
    可视化标签
    """
    img_to_show = cv2.imread(img_path, )
    h0, w0 = img_to_show.shape[0], img_to_show.shape[1]
    
    for anno in annos:
        anno_ = anno.split(' ')
        cls_id = anno_[0]
        track_id = anno_[1]
        xc, yc = float(anno_[2]) * w0, float(anno_[3]) * h0 
        w, h = float(anno_[4]) * w0, float(anno_[5]) * h0 

        x0, y0 = int(xc - w / 2), int(yc - h / 2)
        x1, y1 = int(xc + w / 2), int(yc + h / 2)

        cv2.rectangle(img_to_show, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
        text = f'{cls_id}-{track_id}'
        cv2.putText(img_to_show, text, (x0, y0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, 
                        color=(255, 164, 0), thickness=2)
    
    write_path = osp.join(write_path, img_path.split('/')[-3])
    if not osp.exists(write_path):
        os.makedirs(write_path)
    cv2.imwrite(filename=osp.join(write_path, img_path.split('/')[-1]), img=img_to_show)

if __name__ == '__main__':
    if not osp.exists('./uavdt'):
        os.system('mkdir ./uavdt')
    else:
        os.system('rm -rf ./uavdt/*')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_imgs', action='store_true', help='whether generate soft link of imgs')
    parser.add_argument('--certain_seqs', action='store_true', help='for debug')
    parser.add_argument('--half', action='store_true', help='half frames')
    parser.add_argument('--ratio', type=float, default=0.8, help='ratio of test dataset devide train dataset')
    parser.add_argument('--random', action='store_true', help='random split train and test')

    parser.add_argument('--norm', action='store_true', help='only true when used in yolo training')
    parser.add_argument('--debug', action='store_true')

    opts = parser.parse_args()

    generate_imgs_and_labels(opts)
    # python src/dataset_tools/gen_labels_uavdt.py --generate_imgs --half --debug
    # 
