from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
from PIL import Image
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch

from lib.tracker.multitracker import JDETracker, myTracker
from lib.tracker.depthtracker import Tracker_d
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde as datasets
from lib.datasets.dataset_factory import get_dataset

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} {class_name} -1 -1 -1 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 1\n'
        KITTI_CLS_DICT = {
            1: 'Pedestrian', 
            2: 'Car'
        }
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, classes in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, cls in zip(tlwhs, track_ids, classes):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                if data_type == 'kitti':
                    line = save_format.format(frame=frame_id, id=track_id, class_name=KITTI_CLS_DICT[int(cls)],  x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                else:
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    if 'depth' in opt.task:
        tracker = Tracker_d(opt, frame_rate=frame_rate)

    elif opt.reid:
        tracker = JDETracker(opt, frame_rate=frame_rate)
    else:
        tracker = myTracker(opt, frame_rate=frame_rate)

    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        # online_targets = tracker.update(blob, img0)
        online_targets = tracker.update_depth(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_depths = []
        online_classes = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            cls = t.cls

            if hasattr(t, 'depth'):
                depth = t.depth
            else: depth = None

            # vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_classes.append(cls)
                #online_scores.append(t.score)
                if depth is not None:
                    online_depths.append(depth)

        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids, online_classes))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:

            if 'depth' in opt.task:
                online_im = vis.plot_tracking_depth(img0, online_tlwhs, online_ids, scores=None, depths=online_depths, 
                                        frame_id=frame_id, fps=1. / timer.average_time)
            else:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True, evaluate=False):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'
    # data_type = 'kitti'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(opt, osp.join(data_root, seq, ), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()

        # update opt with dataset
        opt = opts().update_dataset_info_and_set_heads(opt, dataloader)

        frame_rate = 30 
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval        
        if evaluate:
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, data_type)
            accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            images_name = sorted(os.listdir(output_dir))
            img0 = Image.open(os.path.join(output_dir, images_name[0]))
            vw = cv2.VideoWriter(output_video_path, fourcc, 15, img0.size)

            for img in images_name:
                if img.endswith('.jpg'):
                    frame = cv2.imread(os.path.join(output_dir, img))
                    vw.write(frame)
            
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    if evaluate:
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':

    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = '/data/wujiapeng/datasets/MOT17/images/test'
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = '/data/wujiapeng/datasets/MOT17/images/train'
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')

    if opt.test_visdrone:
        seqs_str = '''uav0000009_03358_v
                    uav0000073_00600_v
                    uav0000073_04464_v
                    uav0000077_00720_v
                    uav0000088_00290_v
                    uav0000119_02301_v
                    uav0000120_04775_v
                    uav0000161_00000_v
                    uav0000188_00000_v
                    uav0000201_00000_v
                    uav0000249_00001_v
                    uav0000249_02688_v
                    uav0000297_00000_v
                    uav0000297_02761_v
                    uav0000306_00230_v
                    uav0000355_00001_v
                    uav0000370_00001_v
                '''
        data_root = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019/images/test'
        
    if opt.test_uavdt:
        seqs_str = '''M0101
                M0202
                M0203
                M0402
                M0601
                M0604
                M1002
                M1003
                M1008
                M1202
                '''
        
        data_root = '/data/wujiapeng/datasets/UAVDT/images/test'


    if opt.val_kitti:
        seqs = ['{:04d}'.format(seq) for seq in range(21)]
        data_root = '/data/wujiapeng/datasets/KITTI/training/image_02'
    
    if opt.test_kitti:
        seqs = ['{:04d}'.format(seq) for seq in range(29)]
        data_root = '/data/wujiapeng/datasets/KITTI/testing/image_02'

    if not (opt.val_kitti or opt.test_kitti):
        seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
