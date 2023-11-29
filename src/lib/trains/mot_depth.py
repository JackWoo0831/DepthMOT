from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from progress.bar import Bar

from .base_trainer import BaseTrainer, AverageMeter
from .mot import myMotLoss

from lib.utils.depth_utils import disp_to_depth, transformation_from_parameters, \
                                    BackprojectDepth, Project3D, SSIM, normalize_image

from tensorboardX import SummaryWriter

class MotDepthLoss(nn.Module):
    """
    Loss func with depth (disparity) loss without reid
    TODO
    """
    def __init__(self, opt):
        super().__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()

        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        
        self.opt = opt
        self.device = opt.device

        # for uncertainty loss 
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_depth = nn.Parameter(-1.05 * torch.ones(1))

        h, w = self.opt.img_size[1], self.opt.img_size[0]
        self.h, self.w = h, w 

        # camera intrinsic and its inverse
        self.K = torch.tensor([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).to(self.opt.device)
        self.K[0, :] *= w
        self.K[1, :] *= h 

        self.inv_K = torch.pinverse(self.K).to(self.opt.device)

        self.K = self.K.unsqueeze(0).repeat(self.opt.batch_size, 1, 1)  # (4, 4) -> (bs, 4, 4)
        self.inv_K = self.inv_K.unsqueeze(0).repeat(self.opt.batch_size, 1, 1)  # (4, 4) -> (bs, 4, 4)

        self.scales = 4  # depth scales, see also src/lib/models/networks/dla_depth.py

        # def 2D-3D and 3D-2D projection class
        self.backproject_depth = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
        self.project_3d = Project3D(self.opt.batch_size, h, w).to(self.device)
        
        # def ssim for loss
        self.ssim = SSIM()

        # resize part to gen different scales of input
        self.resize_module = {}
        for i in range(self.scales):
            s = 2 ** i 
            self.resize_module[i] = Resize(size=(h // s, w // s), )

        # TODO: check input format (eg norm, range etc) of fairmot and monodepth
        

    def forward(self, outputs, batch):
        """
        Args:
            outputs: dict, 
            batch: dict, 
        """

        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs  # dict

            # outputs:
            # hm: shape (bs, cls_num, h, w)
            # wh: shape (bs, 4, h, w)
            # id: shape (bs, dim, h, w)
            # reg: shape (bs, 2, h, w)
            
            # batch:
            # hm: shape (bs, cls_num, h, w)
            # reg_mask: shape (bs, K)
            # ind: shape (bs, K)
            # wh: shape (bs, K, 4)
            # reg: shape (bs, K, 2)
            # ids: shape (bs, K)
            # bbox: shape (bs, K, 4)

            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            
            det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
            
            # predict prev and next frame
            self.gen_pred_images(batch, outputs)

            # cal depth loss
            depth_loss = self.compute_depth_loss(batch, outputs)
            depth_loss *= 50
            
            if opt.multi_loss == 'uncertainty':
                loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_depth) * depth_loss + (self.s_det + self.s_depth)
                loss *= 0.5
            else:
                loss = det_loss + depth_loss

            loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss, 'depth_loss': depth_loss * 0.02}
                
        return loss, loss_stats
    
    def gen_pred_images(self, batch, outputs):
        """
        According to estimated camera pose and intrinsic matrix, predict the warraped prev and next image

        """
        
        for scale in range(self.scales):
            # cal loss for every scale
            disp = outputs[f'disp_{scale}']  # torch.Tensor (bs, 1, h, w)
            # to origin model input size 
            disp = F.interpolate(
                input=disp, size=[self.h, self.w], mode='bilinear', align_corners=False
            )

            # source_scale = 0  # default
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            for frame_id in [-1, 1]:  # prev and next

                # axisangle = outputs[f'axis_0_{frame_id}']
                # translation = outputs[f'trans_0_{frame_id}']

                # inv_depth = 1 / depth
                # mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                # transform pose to project map matrix
                T = outputs[f'map_0_{frame_id}']

                cam_points = self.backproject_depth(
                    depth, self.inv_K)
                pix_coords = self.project_3d(
                    cam_points, self.K, T)
                
                outputs[f'color_{frame_id}_{scale}'] = F.grid_sample(
                    batch['prev_input'] if frame_id == -1 else batch['next_input'] , 
                    pix_coords, 
                    padding_mode="border", 
                    align_corners=False
                )

    def gen_current_image(self, prev_img, cur_img, prev_disp, transform, ):
        """
        for debug in inference: pred next images according to current img, disp and camera transform
        """
        _, depth = disp_to_depth(prev_disp, self.opt.min_depth, self.opt.max_depth)
        T = transform

        cam_points = self.backproject_depth(
                    depth, self.inv_K)
        
        pix_coords = self.project_3d(
            cam_points, self.K, T)
        
        if False:
            print(pix_coords.shape)
            print(pix_coords[0, :, 0])  # top left -> bottom left
            print(pix_coords[:, -1, :])  # bottom left -> bottom right 
            print(pix_coords[0, :, -1])  # top right -> bottom right
            print(pix_coords[0, 0, :])  # top left -> top right

                
        ret = F.grid_sample(
            prev_img, 
            pix_coords, 
            padding_mode="border", 
            align_corners=False
        )

        return ret

    def compute_depth_loss(self, batch, outputs):
        """
        depth loss: \mu L_{reprojection} + \lambda L_{smoothness}
        """

        total_loss = 0
        for scale in range(self.scales):
            # for each scale, compute a loss
            loss = 0

            # reprojection loss
            reprojection_loss = []

            target = batch['input']

            # proj: pred -> target
            for frame_id in [-1, 1]:
                pred = outputs[f'color_{frame_id}_{scale}']
                reprojection_loss.append(self._compute_reprojection_loss(pred, target))  # list, length: 2

            reprojection_loss = torch.cat(reprojection_loss, 1)  # torch.Tensor, shape (bs, 2, h, w), h, w is the model input size

            # proj: target -> target
            identity_reprojection_loss = []
            for frame_id in [-1, 1]:
                pred = batch['prev_input'] if frame_id == -1 else batch['next_input'] 
                identity_reprojection_loss.append(
                    self._compute_reprojection_loss(pred, target))

            identity_reprojection_loss = torch.cat(identity_reprojection_loss, 1)

            identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            loss += to_optimise.mean()  # torch.Tensor, shape (1, )

            # smoothness loss
            disp = outputs[f'disp_{scale}']
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            target_scale = self.resize_module[scale](target)

            smooth_loss = self._compute_smooth_loss(norm_disp, target_scale)

            loss += self.opt.smooth_loss_weight * smooth_loss / (2 ** scale)
            total_loss += loss

        total_loss /= self.scales

        return total_loss


    def _compute_reprojection_loss(self, pred, target):
        """
        Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)  # (bs, 3, h, w)
        l1_loss = abs_diff.mean(1, True)  # (bs, 1, h, w)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def _compute_smooth_loss(self, disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()



class myModelWithLoss(nn.Module):
    """
    package the model and loss together
    compared to ModleWithLoss, the model returns a dict 

    """
    def __init__(self, model, loss):
        super().__init__()
        self.model = model 
        self.loss = loss
    def forward(self, batch):
        # TODO: input prev and next frame,
        outputs = self.model(batch['input'], batch['prev_input'], batch['next_input'])  # outputs: dict
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats

class MotDepthTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):

        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = myModelWithLoss(model, self.loss)
        self.optimizer.add_param_group({'params': self.loss.parameters()})

        if not opt.disable_tensorboard:
            self.writer = SummaryWriter(opt.save_dir)
        else:
            self.writer = None 
        
        self.step = 0

    def _get_losses(self, opt):
        if opt.reid:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
            loss = MotDepthLoss(opt)
        else:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
            loss = MotDepthLoss(opt)

        loss_states += ['depth_loss']  # depth loss consists of reproj loss, similarity loss and smoothness loss

        return loss_states, loss
    
    def run_epoch(self, phase, epoch, data_loader):
        """
        rewrite to add log
        """
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    # print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
                    print(Bar.suffix)
            else:
                bar.next()
            
            if opt.test:
                self.save_result(output, batch, results)

            if not self.step % 100:
                self.log(batch, output, loss_stats)

            self.step += 1

            del output, loss, loss_stats, batch
        
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results
    
    def log(self, batch, outputs, loss_stats):
        """
        write to tensorboard
        """

        # show current loss rather than avg loss
        for loss_item, loss_value in loss_stats.items():
            self.writer.add_scalar(tag=f'{loss_item}', scalar_value=loss_value.mean().item(), 
                                   global_step=self.step)
            
        # show predict depth (disparity)
        # show first 4 sample
        for img_idx in range(min(2, self.opt.batch_size)):

            # show input img
            if False:
                self.writer.add_image(
                    tag=f'{img_idx}: prev', 
                    img_tensor=batch['prev_input'][img_idx].data, 
                    global_step=self.step
                )
                self.writer.add_image(
                    tag=f'{img_idx}: current', 
                    img_tensor=batch['input'][img_idx].data, 
                    global_step=self.step
                )
                self.writer.add_image(
                    tag=f'{img_idx}: next', 
                    img_tensor=batch['next_input'][img_idx].data, 
                    global_step=self.step
                )

            # show disparity
            if True:
                for scale_idx in range(4): # default 4 scales
                    self.writer.add_image(
                        tag=f'{img_idx}: scale{scale_idx}', 
                        img_tensor=normalize_image(outputs[f'disp_{scale_idx}'][img_idx]), 
                        global_step=self.step, 
                    )

                    for frame_id in [-1, 1]:
                        self.writer.add_image(
                            tag=f'{img_idx}: pred{frame_id}', 
                            img_tensor=outputs[f'color_{frame_id}_0'][img_idx].data, 
                            global_step=self.step
                        )

                    break              
                



    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]