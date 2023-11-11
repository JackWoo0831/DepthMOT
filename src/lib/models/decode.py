from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = torch.true_divide(topk_inds, width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    # find top K points regardless of class
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # shape: (bs, cls_num, K), topk_inds range: (0, h * w - 1)

    topk_inds = topk_inds % (height * width)
    topk_ys   = torch.true_divide(topk_inds, width).int().float()  # shape: (bs, cls_num, K)
    topk_xs   = (topk_inds % width).int().float()  # shape: (bs, cls_num, K)
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)  # shape: (bs, K), topk_ind range: (0, cls_num * k - 1)
    topk_clses = torch.true_divide(topk_ind, K).int()  # shape: (bs, K), cls id of every top point

    # filter index(topk_inds range: (0, h * w - 1)), xcenter and ycenter
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    """
    Args:
        heat: shape (bs, cls_num, h, w)
        wh: shape (bs, 4, h, w)
        reg: shape: (bs, 2, h, w)
        ltrb: True, K: 500

    """
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds

def mot_depth_decode(heat, wh, depth_map, reg=None, ltrb=False, K=100):
    """
    decode position and depth (disparity) of objects
    Args:
        heat: shape (bs, cls_num, h, w)
        wh: shape (bs, 4, h, w)
        depth_map: shape (bs, 1, h, w)
        reg: shape: (bs, 2, h, w)
        ltrb: True, K: 500

    Return:
        detections: (bs, K, 6)
        depth: (bs, K)
        ind: 
    """
    # TODO 
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    # bbox: (x0, y0, x1, y1) if depth is defined by mean(depth[y1, x0: x1 + 1])
    # first grab the relative rows(x axis)
    bboxes_ = bboxes.clone().long()  # torch.long for index  shape: (bs, K, 4)

    depth = torch.zeros(size=(batch, K), ).to(bboxes.device)

    for batch_idx in range(batch):
        depth_x_axis = depth_map[batch_idx, :, bboxes_[batch_idx, :, 3]]  # shape: (1, K, w)

        start_end_x_axis = bboxes_[batch_idx, :, [0, 2]]  # shape: (K, 2)

        for idx in range(K):
            x0, x1 = start_end_x_axis[idx, 0], start_end_x_axis[idx, 1]

            depth[batch_idx, idx] = depth_x_axis[batch_idx, idx, x0: x1 + 1].mean()

    return detections, depth, inds