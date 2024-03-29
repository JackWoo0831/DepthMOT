import os 
import os.path as osp 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tracking_utils.kalman_filter import KalmanFilter, BotKalmanFilter

from models.decode import mot_depth_decode, mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat

import matplotlib as mpl
import matplotlib.cm as cm 
import PIL.Image as pil

from tracker import matching
from tracker.gmc import GMC

from .basetrack import TrackState
from .multitracker import STrack, JDETracker

from trains.mot_depth import MotDepthLoss
from utils.depth_utils import disp_to_depth

class STrack_d(STrack):
    shared_kalman = BotKalmanFilter()  # Kalman state vector: [xc, yc, w, h, ...]

    def __init__(self, cls, tlwh, score, depth, buffer_size=30):
        # wait activate
        self.cls = cls  # category
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.depth = np.array([round(depth.item() * 1e3), ], dtype=float)[0]
        # self.depth = 2000 - (self._tlwh[1] + self._tlwh[3])  # abl in SparseTrack
        self.depth_update_weight = [0.0, 1]

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.depth = new_track.depth

    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        self.depth = self.depth_update_weight[0] * self.depth \
                         + self.depth_update_weight[1] * new_track.depth
        

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
        
    @staticmethod
    def multi_pmc(stracks, depth_map, K, inv_K, T, comp_pose=False, print_ref=False, iou_check=False):
        """
        Pose motion compensation
        Args:
            stracks: List[STrack_d]
            depth_map: torch.Tensor (h0, w0)
            K: camera intrinsic matrix, torch.Tensor (4, 4)
            inv_K: penrose inv of intrinsic matrix, torch.Tensor (4, 4)
            T: transform matrix of two camera coords in world coord, torch.Tensor (4, 4)
            comp_pose: compensate pixel offset, bool
            print_ref: for debug, bool
            iou_check: check iou for validation, bool
        Return:
            None
        """
        if len(stracks) > 0:
            # step 1, convert kalman mean to tlbr
            multi_mean = np.asarray([st.mean.copy() for st in stracks])[:, :4]  # xywh
            multi_mean[:, :2] -= 0.5 * multi_mean[:, 2:]  # xywh -> tlwh
            multi_mean[:, 2:] += multi_mean[:, :2]  # tlwh -> tlbr

            multi_mean_ori = np.array(multi_mean)  # bkup

            multi_mean_tl = torch.from_numpy(multi_mean[:, :2]).float()
            multi_mean_br = torch.from_numpy(multi_mean[:, 2:]).float()

            multi_mean_bl = torch.vstack([multi_mean_tl[:, 0], multi_mean_br[:, 1]], ).t()  # bottom-left coor

            multi_mean_bl = STrack_d.map_coord(multi_mean_bl, depth_map, K, inv_K, T)
            multi_mean_br = STrack_d.map_coord(multi_mean_br, depth_map, K, inv_K, T)

            # compensate pixel offsets
            if comp_pose:
                
                multi_mean_tl = STrack_d._comp_pose(multi_mean_tl, depth_map, K, inv_K, T, print_ref)
                multi_mean_br = STrack_d._comp_pose(multi_mean_br, depth_map, K, inv_K, T, print_ref)

            # update multi_mean_tl
            multi_mean_tl[:, 0] = multi_mean_bl[:, 0]
            
            multi_mean = torch.cat([multi_mean_tl, multi_mean_br], dim=1).numpy()

            # tlbr -> xywh
            multi_mean[:, :2] = 0.5 * (multi_mean[:, :2] + multi_mean[:, 2:])  # xybr
            multi_mean[:, 2:] = 2 * (multi_mean[:, 2:] - multi_mean[:, :2])


            for i in range(len(stracks)):
               
                if iou_check:
                    # cal iou
                    cur_mean, new_mean = multi_mean_ori[i], multi_mean[i]
                
                    new_mean = STrack.xywh_to_tlbr(new_mean)
                    iou = matching.single_iou(cur_mean, new_mean)

                    if iou >= 0.5:
                        stracks[i].mean[: 4] = multi_mean[i]

                else: 
                    stracks[i].mean[: 4] = multi_mean[i]
                    # stracks[i].mean[: 4] = 0.9 * stracks[i].mean[: 4] + 0.1 * multi_mean[i]

    @staticmethod
    def _comp_pose(coords, depth_map, K, inv_K, T, print_ref=False):
        """
        compensate pixel offset of PoseNet
        """
        h0, w0 = depth_map.shape[0], depth_map.shape[1]
        ref_col = torch.cat([torch.zeros((h0, 1)), torch.arange(0, h0).unsqueeze(1)], dim=1)  # top-left -> bottom-left
        ref_row = torch.cat([torch.arange(0, w0).unsqueeze(1), (h0 - 1) * torch.ones((w0, 1))], dim=1)  # bottom-left -> bottom-right
        
        # solve the new position of ref col and row in current frame
        ref_col = STrack_d.map_coord(ref_col, depth_map, K, inv_K, T)  # look up table, key: y, value: offset of x
        ref_row = STrack_d.map_coord(ref_row, depth_map, K, inv_K, T)  # look up table, key: x, value: offset of y

        if print_ref:
            print(ref_col, ref_row)

        # compensate x
        keys_ = coords[:, 1].long().clamp(0, h0 - 1)
        coords[:, 0] += ref_col[keys_, 0]

        # compensate y
        keys_ = coords[:, 0].long().clamp(0, w0 - 1)
        coords[:, 1] += ref_row[keys_, 1] - h0

        return coords

    @staticmethod
    def map_coord(coords, depth_map, K, inv_K, T):
        """
        map coords between two camera coord system
        Args:
            coords: coordinates, torch.Tensor (N, 2)
            K, inv_K, T: same as multi_gmc()
        Return:
            np.ndarray (N, 2)
        """
        N = coords.shape[0]
        h, w = depth_map.shape[0], depth_map.shape[1]

        _, depth_map = disp_to_depth(depth_map, 0.1, 500)
        
        coords = torch.cat([coords.t(), torch.ones((1, N))], dim=0)  # (3, N)

        # cam -> world
        cam_points = torch.mm(inv_K[:3, :3], coords)  # (3, N)

        # add depth
        ind_row = torch.clamp(coords[1, :].long(), 0, h - 1)
        ind_col = torch.clamp(coords[0, :].long(), 0, w - 1)

        depth = depth_map[ind_row, ind_col]  # (N, )

        cam_points = depth.view(1, -1) * cam_points

        # make dim as 4
        cam_points = torch.cat([cam_points, torch.ones((1, N))], dim=0)  # (4, N)

        # world -> cam
        P = torch.mm(K, T)[:3, :]  # (3, 4)

        cam_points = torch.mm(P, cam_points)  # (3, N)

        pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-7)  # (2, N)

        return pix_coords.t()
                
    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret
    
    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

class Tracker_d(JDETracker):

    def __init__(self, opt, frame_rate=30):
        super().__init__(opt, frame_rate)

        self.debug = False

        if self.opt.motion_comp == 'none':
            self.motion_comp = None 
        elif self.opt.motion_comp == 'bot':
            self.motion_comp = GMC(method='orb', )
        else:
            self.motion_comp = None 

        if self.debug:
            self.loss = MotDepthLoss(opt)
        else:
            self.loss = None 

        self.kalman_filter = BotKalmanFilter()

        # intrinsic matrix
        h, w = self.opt.img_size[1], self.opt.img_size[0]
        self.K = torch.tensor([[0.6188, 0, 0.5, 0],
                            [0, 1.8957, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])  # visdrone
        
        self.K = torch.tensor([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])  # uavdt and kitti

    def _get_depth(self, depth_map, bboxes, h0=608, w0=1088):
        """
        bbox: (x0, y0, x1, y1)
        depth is defined by mean(depth[y1, x0: x1 + 1])
        
        Args:
            depth_map: torch.Tensor, (h0, w0)
            bboxes: np.ndarray, (num of objs, 6)
            h0, w0: origin size
        """

        bboxes_ = torch.from_numpy(bboxes[:, :4])
        bboxes_ = bboxes_.long()

        depth = torch.zeros((bboxes.shape[0], ))

        for idx in range(bboxes.shape[0]):

            x0, y0, x1, y1 = bboxes_[idx]

            # restrict ranges
            x0 = torch.maximum(x0, torch.tensor([0], dtype=torch.long))
            y1 = torch.minimum(y1, torch.tensor([h0 - 1], dtype=torch.long))

            depth[idx] = depth_map[y1, x0: x1 + 1].mean()

        return depth.numpy()
    
    def _draw_depth_map(self, depth_map):
        """
        visualize the depth map
        """
        vmax = np.percentile(depth_map, 95)
        normalizer = mpl.colors.Normalize(vmin=depth_map.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(depth_map)[:, :, :3] * 255).astype(np.uint8)

        im = pil.fromarray(colormapped_im)
        im.save(f'./debug_imgs/test_{self.frame_id}.jpg')

    def _get_deep_range(self, obj, step):
        col = []
        for t in obj:
            lend = t.depth
            col.append(lend)

        max_len, min_len = max(col), min(col)
        if max_len != min_len:
            deep_range =np.arange(min_len, max_len, (max_len - min_len + 1) / step)  # + 1
            if deep_range[-1] < max_len:
                deep_range = np.concatenate([deep_range, np.array([max_len],)])
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:    
            deep_range = [min_len, ]
        mask = self._get_sub_mask(deep_range, col)      
        return mask
    
    def _get_sub_mask(self, deep_range, col):
        min_len=deep_range[0]
        max_len=deep_range[-1]
        if max_len == min_len:
            lc = min_len   
        mask = []

        for d in deep_range:
            if d > deep_range[0] and d < deep_range[-1]:
                mask.append((col >= lc) & (col < d)) 
                lc = d
            elif d == deep_range[-1]:
                mask.append((col >= lc) & (col <= d)) 
                lc = d 
            else:
                lc = d
                continue
        return mask
    
    def depth_cascade_matching(self, detections, tracks, activated_starcks, refind_stracks, levels, thresh, is_fuse):
        if len(detections) > 0:
            det_mask = self._get_deep_range(detections, levels) 
        else:
            det_mask = []

        if len(tracks)!=0:
            track_mask = self._get_deep_range(tracks, levels)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if  len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])
        
            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)
                
                # search det 
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks
                
                dists = matching.iou_distance(track_, det_)
               
                matches, u_track_, u_det_ = matching.linear_assignment(dists, thresh)
                for itracked, idet in matches:
                    track = track_[itracked]
                    det = det_[idet]
                    if track.state == TrackState.Tracked:
                        track.update(det_[idet], self.frame_id)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
                u_tracks = [track_[t] for t in u_track_]
                u_detection = [det_[t] for t in u_det_]
                
            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections
            
        return activated_starcks, refind_stracks, u_tracks, u_detection
    
    def _pred_images(self, prev_img, cur_img, prev_disp, transform, ):
        """
        for debug: pred next images according to current img, disp and camera transform
        """
        if self.loss is not None:
            print(transform.shape)
            return self.loss.gen_current_image(prev_img, cur_img, prev_disp, transform, )
        else:
            return None
        
    def _plot_det(self, img, dets, dets2=None, ):
        """
        for debug: plot detections in img. dets: tlbr
        """
        for idx, det in enumerate(dets):
            x0, y0, x1, y1 = det

            intbox = tuple(map(int, (x0, y0, x1, y1)))
            cv2.rectangle(img, intbox[0:2], intbox[2:4], color=(255, 0, 0), thickness=1)

        if dets2 is not None:
            for idx, det in enumerate(dets2):
                x0, y0, x1, y1 = det

                intbox = tuple(map(int, (x0, y0, x1, y1)))
                cv2.rectangle(img, intbox[0:2], intbox[2:4], color=(0, 255, 0), thickness=1)

        cv2.imwrite(f'debug_imgs/det_frame{self.frame_id}.jpg', img)
    
    def update_depth(self, im_blob, img0):
        self.frame_id += 1
        if self.frame_id == 1:
            self.prev_img = None  # previous image for motion compensation, None | torch.Tensor, shape (1, 3, input_h, input_w)
            self.prev_depth_map = None  # previous depth map for motion compensation, None | torch.Tensor, shape (1, 1, input_h, input_w)
            self.prev_depth_map_ori = None 

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        
        # update intrinsic matrix
        if self.frame_id == 1:
            self.K_ = self.K.clone()
            self.K_[0, :] *= width 
            self.K_[1, :] *= height 

            self.inv_K_ = torch.pinverse(self.K_)

        ''' Step 1: Network forward, get detections'''
        with torch.no_grad():
            output = self.model.inference(im_blob, self.prev_img)
            hm = output['hm'].sigmoid_()  # shape: (bs, cls_num, h, w)
            wh = output['wh']  # shape: (bs, 4, h, w)
            depth_map = output['disp_0']  # shape: (bs, 1, h, w)
            transformation = output['map_-1_0']  # shape: (bs, 4, 4)
            if transformation is not None:
                transformation = transformation.squeeze().cpu()

            # interpolate depth map under (inputh, inputw) to (h0, w0)
            depth_map_ori = F.interpolate(depth_map, size=(height, width), mode='nearest')  # (bs, 1, h0, w0)
            depth_map_ori = depth_map_ori[0, 0].cpu()  # (h0, w0)

            # assert not torch.isnan(depth_map_ori).any().item(), 'exists nan!'

            reg = output['reg'] if self.opt.reg_offset else None  # shape: (bs, 2, h, w)
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            # dets, depths, inds = mot_depth_decode(hm, wh, depth_map=depth_map, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)  # TODO: decode depth here
            # depths = depths[0]

        # debug
        # if False:
        if self.debug and self.prev_depth_map is not None:
            # ret = self._pred_images(self.prev_img, im_blob, self.prev_depth_map, transformation.cuda())
            ret = self._pred_images(self.prev_img, im_blob, self.prev_depth_map, transformation.cuda())
            ret = ret.permute(0, 2, 3, 1)
            ret = ret.squeeze().cpu().numpy()
            ret = ret * 255
            ret = ret.astype(np.uint8)            
            im = pil.fromarray(ret)
            im.save(f'debug_imgs/pred_{self.frame_id}.jpg')

        dets_dict = self.post_process(dets, meta)  # dict key: cls_id value: np.ndarray (obj num, 5)
        # merge all categories
        dets_list = []
        for cls_id, boxes in dets_dict.items():
            cls_array = cls_id * np.ones((boxes.shape[0], 1))
            dets_list.append(np.hstack((boxes, cls_array)))
        
        dets = np.concatenate(dets_list, axis=0)  # shape: (obj num, 6)

        # byte-track style
        inds_high = dets[:, 4] > self.opt.conf_thres
        inds_low = dets[:, 4] > 0.1
        inds_second = np.logical_and(inds_low, np.logical_not(inds_high))

        dets_first = dets[inds_high]
        dets_second = dets[inds_second]

        # get depth corresponding to dets
        depth_first = self._get_depth(depth_map_ori, dets_first, h0=height, w0=width)
        depth_second = self._get_depth(depth_map_ori, dets_second, h0=height, w0=width)

        
        # init high-score dets
        if len(depth_first) > 0:
            '''Detections'''
            detections = [STrack_d(cls, STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], depth, 30) for
                          (cls, tlbrs, depth) in zip(dets_first[:, 5], dets_first[:, :5], depth_first[:, ])]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        STrack.multi_predict(strack_pool)

        
        # motion compensation
        if self.opt.motion_comp == 'bot':
            warp = self.motion_comp.apply(img0, dets)
            STrack_d.multi_gmc(strack_pool, warp)
            STrack_d.multi_gmc(unconfirmed, warp)
        elif self.opt.motion_comp == 'pose':
            # kalman: predict position in current coordinate axis -> transform coordinate axis
            STrack_d.multi_pmc(strack_pool, self.prev_depth_map_ori, self.K_, self.inv_K_, transformation)
            STrack_d.multi_pmc(unconfirmed, self.prev_depth_map_ori, self.K_, self.inv_K_, transformation)

        activated_starcks, refind_stracks, u_track, u_detection_high = self.depth_cascade_matching(
                                                                                detections, 
                                                                                strack_pool, 
                                                                                activated_starcks,
                                                                                refind_stracks, 
                                                                                levels=1, 
                                                                                thresh=0.75, 
                                                                                is_fuse=True)  

        # debug
        if False:
            img0_ = np.array(img0)
            
            if self.frame_id == 1:
                self._plot_det(img0_, dets_first[:, :4])
                self.prev_det = torch.from_numpy(dets_first[:, :4]).float()
            
            else:               

                # t = transformation.repeat(self.prev_det.shape[0], 1, 1).cpu()
                # det_cur = torch.bmm(t.float(), self.prev_det.unsqueeze(2).float()).numpy()

                # kalman
                multi_mean = np.asarray([st.mean.copy() for st in strack_pool])[:, :4]  # xywh
            
                multi_mean[:, :2] -= 0.5 * multi_mean[:, 2:]  # xywh -> tlwh
                multi_mean[:, 2:] += multi_mean[:, :2]  # tlwh -> tlbr

                self._plot_det(img0_, multi_mean)

                # self._draw_depth_map(depth_map_ori)

                if self.frame_id == 60: exit()


        ''' Step 3: Second association, with low score detections'''

        if len(dets_second) > 0:
            detections_second = [STrack_d(cls, STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], depth, 30) for
                          (cls, tlbrs, depth) in zip(dets_second[:, 5], dets_second[:, :5], depth_second[:, ])]
        else:
            detections_second = []

        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]   

        activated_starcks, refind_stracks, u_track, u_detection_sec = self.depth_cascade_matching(
                                                                                detections_second, 
                                                                                r_tracked_stracks, 
                                                                                activated_starcks, 
                                                                                refind_stracks, 
                                                                                levels=1, 
                                                                                thresh=0.3, 
                                                                                is_fuse=False) 

        for track in u_track:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)  

        
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame 
        detections = [d for d in u_detection_high ]
        dists = matching.iou_distance(unconfirmed, detections)
    
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.8) 
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        self.prev_img = im_blob
        self.prev_img_ori = torch.from_numpy(img0).unsqueeze(0)
        self.prev_depth_map = depth_map
        self.prev_depth_map_ori = depth_map_ori

        return output_stracks
    

def joint_stracks(tlista, tlistb):

    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb