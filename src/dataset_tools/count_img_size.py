import cv2 
import os 
import numpy as np 
from collections import defaultdict

def count_img_size(data_root, dataset_name):

    if dataset_name == 'visdrone':
        seq_path = '{data_root}/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/sequences'
        img_format = '{data_root}/VisDrone2019/VisDrone2019/VisDrone2019-MOT-train/sequences/{seq_name}'
    elif dataset_name == 'uavdt':
        seq_path = '{data_root}/UAVDT/UAV-benchmark-M'
        img_format = '{data_root}/UAVDT/UAV-benchmark-M/{seq_name}/img1/'
    else:
        raise NotImplementedError
    
    size_dict = defaultdict(int)
    
    seq_list = os.listdir(seq_path.format(data_root=data_root))

    for seq in seq_list:
        img_list = os.listdir(img_format.format(data_root=data_root, seq_name=seq))
        img_path = os.path.join(img_format.format(data_root=data_root, seq_name=seq), img_list[0])

        img_eg = cv2.imread(img_path)

        assert img_eg is not None, f'{img_path}'

        h, w = img_eg.shape[0], img_eg.shape[1]
        key_hash = f'{h} {w}'

        size_dict[key_hash] += 1

    avg_h, avg_w = 0, 0

    for k, v in size_dict.items():
        proportion = v / len(seq_list)
        k_ = [int(k.split(' ')[0]), int(k.split(' ')[1])]
        print(f'{k_}: {proportion * 100} %')

        avg_h += k_[0] * proportion
        avg_w += k_[1] * proportion

    print(f'Avg size: {avg_h}, {avg_w}')

if __name__ == '__main__':

    data_root = '/data/wujiapeng/datasets'
    dataset_name = 'uavdt'

    count_img_size(data_root, dataset_name)
    
    """
    result:
    visdrone:
    [1512, 2688]: 19.642857142857142 %
    [1071, 1904]: 53.57142857142857 %
    [756, 1344]: 19.642857142857142 %
    [1080, 1920]: 1.7857142857142856 %
    [1530, 2720]: 3.571428571428571 %
    [765, 1360]: 1.7857142857142856 %
    Avg size: 1106.8392857142856, 1967.7142857142856

    uavdt:
    [540, 1024]: 98.0 %
    [540, 960]: 2.0 %
    Avg size: 540.0, 1022.72
    """