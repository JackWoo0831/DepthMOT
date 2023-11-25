"""
产生yolo格式的注释文件 VisDrone2019
本代码改动为产生的注释文件具有【连续的ID】 便于DarkNet类的self.nID计算
例如 'VisDrone/labels_with_ids/train/uav0000076_00720_v/000010.txt'
"""

import os
import os.path as osp
import argparse
import cv2
import numpy as np 

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'

CERTAIN_SEQS = []
IGNORE_SEQS = []

VALID_CLASS = [1, 4, 5, 6, 9]  # valid class id
VALID_CLASS_DICT = {cls_id: idx for idx, cls_id in enumerate(VALID_CLASS)}
VALID_CLASS_DICT = {1: 0, 4: 1, 5: 1, 6: 1, 9: 1}  # merge cars

image_wh_dict = {}  # seq->(w,h) 字典 用于归一化

save_split_dict = {  # 储存img和label时候的split
    'VisDrone2019-MOT-train': 'train', 
    'VisDrone2019-MOT-val': 'val', 
    'VisDrone2019-MOT-test-dev': 'test', 
}

def generate_imgs(split='VisDrone2019-MOT-train', certain_seqs=False, write_txt_path='', half=False):
    """
    产生图片文件夹 例如 VisDrone/images/VisDrone2019-MOT-train/uav0000076_00720_v/000010.jpg
    同时产生序列->高,宽的字典 便于后续

    split: str, 'VisDrone2019-MOT-train', 'VisDrone2019-MOT-val' or 'VisDrone2019-MOT-test-dev'
    CERTAIN_SEQS: bool, use for debug. 
    write_txt_path: path to write path txt file
    """

    if not certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split, 'sequences'))  # 所有序列名称
    else:
        seq_list = CERTAIN_SEQS

    seq_list = [seq for seq in seq_list if seq not in IGNORE_SEQS]

    if osp.exists(osp.join(write_txt_path, './visdrone.txt')):
        os.system(f"rm {osp.join(write_txt_path, './visdrone.txt')}")

    with open(osp.join(write_txt_path, './visdrone.txt'), 'a') as f:
        for seq in seq_list:
            print(f'generate images, processing seq {seq}')
            img_dir = osp.join(DATA_ROOT, split, 'sequences', seq)  # 该序列下所有图片路径 

            imgs = sorted(os.listdir(img_dir))  # 所有图片
            if half: imgs = imgs[: len(imgs) // 2]

            to_path = osp.join(DATA_ROOT, 'images', save_split_dict[split], seq)  # 该序列图片存储位置
            if not osp.exists(to_path):
                os.makedirs(to_path)

            for img in imgs:  # 遍历该序列下的图片

                f.write('VisDrone2019/' + 'VisDrone2019/' + 'images/' + save_split_dict[split] + '/' \
                        + seq + '/' + img + '\n')

                os.symlink(osp.join(img_dir, img),
                            osp.join(to_path, img))  # 创建软链接

            img_sample = cv2.imread(osp.join(img_dir, imgs[0]))  # 每个序列第一张图片 用于获取w, h
            w, h = img_sample.shape[1], img_sample.shape[0]  # w, h

            image_wh_dict[seq] = (w, h)  # 更新seq->(w,h) 字典
    f.close()


def generate_labels(split='VisDrone2019-MOT-train', certain_seqs=False):
    """
    split: str, 'train', 'val' or 'test'
    CERTAIN_SEQS: bool, use for debug. 
    """
    # from choose_anchors import image_wh_dict
    # print(image_wh_dict)
    if not certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split, 'sequences'))  # 序列列表
    else:
        seq_list = CERTAIN_SEQS

    seq_list = [seq for seq in seq_list if seq not in IGNORE_SEQS]

    # 每张图片分配一个txt
    # 要从sequence的txt里分出来

    current_id = 0  # 表示写入txt的当前id 一个目标在所有视频序列里是唯一的 
    last_id = -1  # 表示写入的上一个id
    # 注意 这种方式只适用于真值文件是按目标id排列的 如果按帧序数排列会出错
    # 现在可以保证certain_seqs都符合按id排列


    for seq in seq_list:
        seq_dir = osp.join(DATA_ROOT, split, 'annotations', seq + '.txt')  # 真值文件
        with open(seq_dir, 'r') as f:
            lines = f.readlines()

            for row in lines:
                current_line = row.split(',')  # 读取gt的当前行

                # 不满足特定类就略过
                if (current_line[6] == '0') or (current_line[7] not in ['1', '4', '5', '6', '9']):
                    continue

                # 需要写进新文件行的文字
                frame = current_line[0]  # 第几帧
                id_in_frame = int(current_line[1])  # 当前帧当前目标的id

                if not id_in_frame == last_id:  # 如果到了下一个id 
                    current_id += 1  # 写入文件的id加一
                    last_id = id_in_frame  # 更新last id

                # 写到对应图片的txt

                
                # to_file = osp.join(DATA_ROOT, 'labels_with_ids', split, seq, 'img1', frame.zfill(6) + '.txt')
                to_file = osp.join(DATA_ROOT, 'labels_with_ids', split, seq)
                if not osp.exists(to_file):
                    os.makedirs(to_file)

                to_file = osp.join(to_file, frame.zfill(7) + '.txt')
                with open(to_file, 'a') as f_to:
                    x0, y0 = int(current_line[2]), int(current_line[3])  # 左上角 x y
                    w, h = int(current_line[4]), int(current_line[5])  # 宽 高

                    x_c, y_c = x0 + w // 2, y0 + h // 2  # 中心点 x y

                    image_w, image_h = image_wh_dict[seq][0], image_wh_dict[seq][1]  # 图像高宽
                    # 归一化
                    w, h = w / image_w, h / image_h
                    x_c, y_c = x_c / image_w, y_c / image_h

                    write_line = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        current_id, x_c, y_c, w, h)

                    f_to.write(write_line)
                f_to.close()
        f.close()

    print(f"Total IDs{current_id}")


def generate_labels_multi_cls(split='VisDrone2019-MOT-train', certain_seqs=False, debug=False, half=False):
    """
    针对multi-class真值标签生成, 为了训练效果, 每个cls的track_id都单独处理

    """

    if not certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split, 'sequences'))  # 序列列表
    else:
        seq_list = CERTAIN_SEQS

    seq_list = [seq for seq in seq_list if seq not in IGNORE_SEQS]

    # 记录当前seq的每个类别的id offset 即应该从哪个id开始
    start_id_in_seq = {cls_id: 0 for cls_id in VALID_CLASS}


    for seq in seq_list:
        print(f'generate labels, processing {seq}')
        seq_anno_path = osp.join(DATA_ROOT, split, 'annotations', seq + '.txt')  # 真值文件
        seq_anno = np.loadtxt(seq_anno_path, dtype=float, delimiter=',')

        imgs_name = sorted(os.listdir(os.path.join(DATA_ROOT, split, 'sequences', seq)))

        if half: imgs_name = imgs_name[: len(imgs_name) // 2]

        # 需要预处理出当前序列 每个类别的标注id: 训练需要的id的映射
        seq_id_map = {cls_id: set() for cls_id in VALID_CLASS}
        for cls_id in VALID_CLASS:
            seq_cls_anno = seq_anno[seq_anno[:, 7] == float(cls_id), :]

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


        # 处理每一帧
        for idx, img in enumerate(imgs_name):

            # 读取当前帧的anno
            frame_anno = seq_anno[seq_anno[:, 0] == float(idx + 1), :]

            # 要写入txt的内容
            write_lines = []

            for row_idx in range(frame_anno.shape[0]):
                
                cls_id = int(frame_anno[row_idx, 7])

                if int(frame_anno[row_idx, 6]) == 1 and cls_id in VALID_CLASS:  # 如果有效
                    
                    frame_id = int(frame_anno[row_idx, 0])
                    track_id = int(frame_anno[row_idx, 1])

                    x0, y0 = int(frame_anno[row_idx, 2]), int(frame_anno[row_idx, 3])
                    w, h = int(frame_anno[row_idx, 4]), int(frame_anno[row_idx, 5])

                    xc, yc = x0 + w * 0.5, y0 + h * 0.5

                    # 坐标归一化
                    image_w, image_h = image_wh_dict[seq][0], image_wh_dict[seq][1]  # 图像高宽

                    xc_norm, yc_norm = xc / image_w, yc / image_h
                    w_norm, h_norm = w / image_w, h / image_h

                    # 计算应该填入标注的track id, 为当前seq的索引值+该seq的索引offset+1
                    track_id_ = seq_id_map_[cls_id].index(track_id) + start_id_in_seq[cls_id] + 1

                    write_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        VALID_CLASS_DICT[cls_id], track_id_, xc_norm, yc_norm, w_norm, h_norm)
                    
                    write_lines.append(write_str)

            # 写入txt
            to_file = osp.join(DATA_ROOT, 'labels_with_ids', save_split_dict[split], seq)
            if not osp.exists(to_file):
                os.makedirs(to_file)

            to_file = osp.join(to_file, str(idx + 1).zfill(7) + '.txt')

            with open(to_file, 'a') as f_to:
                for line in write_lines:
                    f_to.write(line)

            # debug 标签可视化
            if debug and idx < 30:  # 每个序列可视化前30帧
                vis_labels(img_path=osp.join(DATA_ROOT, split, 'sequences', seq, img), annos=write_lines)

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
    
    write_path = osp.join(write_path, img_path.split('/')[-2])
    if not osp.exists(write_path):
        os.makedirs(write_path)
    cv2.imwrite(filename=osp.join(write_path, img_path.split('/')[-1]), img=img_to_show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='VisDrone2019-MOT-val', help='train or test')
    parser.add_argument('--write_txt_path', type=str, default='./src/data')
    parser.add_argument('--certain_seqs', type=bool, default=False, help='for debug')
    parser.add_argument('--multi_cls', action='store_true', help='multi class label')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--half', action='store_true')

    opt = parser.parse_args()

    generate_imgs(opt.split, opt.certain_seqs, write_txt_path=opt.write_txt_path, half=opt.half)

    if opt.multi_cls:
        generate_labels_multi_cls(opt.split, certain_seqs=opt.certain_seqs, debug=opt.debug, half=opt.half)
    else:
        generate_labels(opt.split, certain_seqs=opt.certain_seqs)

    print('Done!')

    # python src/dataset_tools/gen_labels_visdrone.py --split VisDrone2019-MOT-train --multi_cls --debug
    # python src/dataset_tools/gen_labels_visdrone.py --split VisDrone2019-MOT-val --multi_cls --debug
    # python src/dataset_tools/gen_labels_visdrone.py --split VisDrone2019-MOT-test-dev --multi_cls --debug