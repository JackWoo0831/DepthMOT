"""
gen labels with FairMOT format on KITTI dataset
"""

import os
import os.path as osp
import argparse
import cv2
import numpy as np 

DATA_ROOT = '/data/wujiapeng/datasets/KITTI'

VALID_CLASS = ['Pedestrian', 'Car']
VALID_CLASS_DICT = {cls_name: idx for idx, cls_name in enumerate(VALID_CLASS)}

image_wh_dict = {}

save_split_dict = {
    'training': 'train', 
    'testing': 'test', 
}

def generate_imgs(split='training', half=False, write_txt_path=''):

    seq_list = os.listdir(osp.join(DATA_ROOT, split, 'image_02'))

    if osp.exists(osp.join(write_txt_path, './kitti.txt')):
        os.system(f"rm {osp.join(write_txt_path, './kitti.txt')}")

    with open(osp.join(write_txt_path, './kitti.txt'), 'a') as f:
        for seq in seq_list:
            print(f'generate images, processing seq {seq}')
            img_dir = osp.join(DATA_ROOT, split, 'image_02', seq)  # 该序列下所有图片路径 

            imgs = sorted(os.listdir(img_dir))  # 所有图片
            if half: imgs = imgs[: len(imgs) // 2]

            to_path = osp.join(DATA_ROOT, 'images', save_split_dict[split], seq)  # 该序列图片存储位置
            if not osp.exists(to_path):
                os.makedirs(to_path)

            for img in imgs:  # 遍历该序列下的图片

                f.write('KITTI/' + 'images/' + save_split_dict[split] + '/' \
                        + seq + '/' + img + '\n')

                os.symlink(osp.join(img_dir, img),
                            osp.join(to_path, img))  # 创建软链接

            img_sample = cv2.imread(osp.join(img_dir, imgs[0]))  # 每个序列第一张图片 用于获取w, h
            w, h = img_sample.shape[1], img_sample.shape[0]  # w, h

            image_wh_dict[seq] = (w, h)  # 更新seq->(w,h) 字典
    f.close()


def generate_labels_multi_cls(split='training', certain_seqs=False, debug=False, half=False):
   

    seq_list = os.listdir(osp.join(DATA_ROOT, split, 'image_02'))

    # 记录当前seq的每个类别的id offset 即应该从哪个id开始
    start_id_in_seq = {cls_id: 0 for cls_id, _ in enumerate(VALID_CLASS)}


    for seq in seq_list:
        print(f'generate labels, processing {seq}')
        seq_anno_path = osp.join(DATA_ROOT, split, 'label_02', seq + '.txt')  # 真值文件

        current_seq_annos = []  # 存储成一般的格式 方便直接转成numpy

        with open(seq_anno_path, 'r') as f:

            for line in f:
                
                line = line.strip()
                # care pos 0, 1, 2, 6, 7, 8, 9
                elements = line.split(' ')

                frame_id = float(elements[0])
                obj_id = float(elements[1])
                cls_name = elements[2]

                if cls_name in VALID_CLASS:
                    cls_id = float(VALID_CLASS_DICT[cls_name])

                    x0, y0, x1, y1 = list(map(float, elements[6: 10]))
                    current_seq_annos.append([frame_id, obj_id, x0, y0, x1, y1, cls_id])

        
        seq_anno = np.array(current_seq_annos, dtype=float)

        imgs_name = sorted(os.listdir(os.path.join(DATA_ROOT, split, 'image_02', seq)))

        if half: imgs_name = imgs_name[: len(imgs_name) // 2]

        # 需要预处理出当前序列 每个类别的标注id: 训练需要的id的映射
        seq_id_map = {cls_id: set() for cls_id, _ in enumerate(VALID_CLASS)}
        for cls_id, cls_name in enumerate(VALID_CLASS):
            seq_cls_anno = seq_anno[seq_anno[:, 6] == float(cls_id), :]

            for row_idx in range(seq_cls_anno.shape[0]):
                seq_id_map[cls_id].add(seq_cls_anno[row_idx, 1])  # set 自动去重

        # 将seq_id_map的set改为list 方便后面索引
        seq_id_map_ = {cls_id: list() for cls_id, _ in enumerate(VALID_CLASS)}
        for k, v in seq_id_map.items():
            seq_id_map_[k] = sorted(list(v))

        # 储存每个序列中每个类别的最大ID号是多少
        # 这样在进行下一个序列遍历的时候 ID在此基础上累加
        max_id_in_seq = {cls_id: 0 for cls_id, _ in enumerate(VALID_CLASS)}
        for key in seq_id_map.keys():
            max_id_in_seq[key] = len(seq_id_map[key])

        # 打印当前seq的id信息
        print(f'===In seq {seq}, id info:')
        for cls_id, _ in enumerate(VALID_CLASS):
            print(f'===cls: {cls_id}, start_id: {start_id_in_seq[cls_id]}, max_id: {max_id_in_seq[cls_id]}')


        # 处理每一帧
        for idx, img in enumerate(imgs_name):

            # 读取当前帧的anno
            frame_anno = seq_anno[seq_anno[:, 0] == float(idx + 1), :]

            # 要写入txt的内容
            write_lines = []

            for row_idx in range(frame_anno.shape[0]):
                
                cls_id = int(frame_anno[row_idx, 6])
                    
                frame_id = int(frame_anno[row_idx, 0])
                track_id = int(frame_anno[row_idx, 1])

                x0, y0 = int(frame_anno[row_idx, 2]), int(frame_anno[row_idx, 3])
                x1, y1 = int(frame_anno[row_idx, 4]), int(frame_anno[row_idx, 5])

                w, h = x1 - x0, y1 - y0

                xc, yc = x0 + w * 0.5, y0 + h * 0.5

                # 坐标归一化
                image_w, image_h = image_wh_dict[seq][0], image_wh_dict[seq][1]  # 图像高宽

                xc_norm, yc_norm = xc / image_w, yc / image_h
                w_norm, h_norm = w / image_w, h / image_h

                # 计算应该填入标注的track id, 为当前seq的索引值+该seq的索引offset+1
                track_id_ = seq_id_map_[cls_id].index(track_id) + start_id_in_seq[cls_id] + 1

                write_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    cls_id, track_id_, xc_norm, yc_norm, w_norm, h_norm)
                
                write_lines.append(write_str)

            # 写入txt
            to_file = osp.join(DATA_ROOT, 'labels_with_ids', save_split_dict[split], seq)
            if not osp.exists(to_file):
                os.makedirs(to_file)

            to_file = osp.join(to_file, str(idx).zfill(6) + '.txt')

            with open(to_file, 'a') as f_to:
                for line in write_lines:
                    f_to.write(line)

            # debug 标签可视化
            if debug and idx < 30:  # 每个序列可视化前30帧
                vis_labels(img_path=osp.join(DATA_ROOT, split, 'image_02', seq, img), annos=write_lines)

        # 更新id offset
        for cls_id, _ in enumerate(VALID_CLASS):
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
    parser.add_argument('--split', type=str, default='training', help='train or test')
    parser.add_argument('--write_txt_path', type=str, default='./src/data')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--half', action='store_true')

    opt = parser.parse_args()

    generate_imgs(opt.split, half=opt.half, write_txt_path=opt.write_txt_path, )

    generate_labels_multi_cls(opt.split, certain_seqs=False, debug=opt.debug, half=opt.half)

    print('Done!')

    # python src/dataset_tools/gen_labels_kitti.py --split training
    # python src/dataset_tools/gen_labels_kitti.py --split testing