import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
import numpy as np

from open3d.ml.torch.ops import voxelize, ragged_to_dense

from .base_model_objdet import BaseModel

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ...datasets.utils import BEVBox3D
from ...datasets.utils import LaserScan, SemLaserScan
from ...datasets.augment import ObjdetAugmentation

import __init__ as booger

#torch.autograd.set_detect_anomaly(True)

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class mtSalsaNext(BaseModel):

    def __init__(self,
                 name="mtSalsaNext",
                 device="cuda",
                 od_classes=['pedestrian', 'bike', 'car', 'other_vehicle'],
                 od_num_classes=4,
                 ss_classes=['pedestrian', 'bike', 'car', 'other_vehicle', 'driveable_surface', 'sidewalk', 'terrain', 'manmade', 'vegetation'],
                 ss_num_classes=9,
                 ignore_label_inds=[-1],
                 num_per_class=[46687, 7834, 501416, 410507, 4069879, 746905, 696526, 2067585, 1565272],
                 sensor_fov_up=10,
                 sensor_fov_down=-30,
                 sensor_img_width=2048,
                 sensor_img_height=64,
                 sensor_img_means=[10.,  0.,  0., -1., 0.2],
                 sensor_img_stds =[10., 10., 10.,  1., 0.2],
                 max_points=150000,
                 head={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         device=device,
                         **kwargs)

        self.od_classes = od_classes
        self.od_num_classes = od_num_classes
        self.od_name2lbl = {n: i for i, n in enumerate(od_classes)}
        self.od_lbl2name = {i: n for i, n in enumerate(od_classes)}
        self.ss_classes = ss_classes
        self.ss_num_classes = ss_num_classes
        self.ss_name2lbl = {n: i for i, n in enumerate(ss_classes)}
        self.ss_lbl2name = {i: n for i, n in enumerate(ss_classes)}
        self.ignore_label_inds = ignore_label_inds
        num_per_class = np.array(num_per_class, dtype=np.float32)
        class_weight = num_per_class / float(sum(num_per_class))
        self.ss_class_weight = torch.tensor(1/(class_weight+0.02)).to(device)

        self.sensor_img_H = sensor_img_height
        self.sensor_img_W = sensor_img_width
        self.sensor_img_means = torch.tensor(sensor_img_means,
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor_img_stds,
                                            dtype=torch.float)
        self.sensor_fov_up = sensor_fov_up
        self.sensor_fov_down = sensor_fov_down
        self.max_points = max_points

        self.anchor_sizes = head.get("sizes", {})
        self.nms_pre = head.get("nms_pre", {})
        self.score_thr = head.get("score_thr", {})
        self.dir_offset = head.get("dir_offset", {})
        self.iou_thr = head.get("iou_thr", {})
        if len(self.iou_thr) != od_num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * od_num_classes
        assert len(self.iou_thr) == od_num_classes

        self.do_transform = False
        #self.augmenter = ObjdetAugmentation(self.cfg.augment, seed=self.rng)

        self.downCntx = ResContextBlock(5, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.conv_cls = nn.Conv2d(32, ss_num_classes, kernel_size=3, padding=1)
        self.conv_reg = nn.Conv2d(32, 7, kernel_size=3, padding=1)
        self.conv_dir = nn.Conv2d(32, 2, kernel_size=3, padding=1)

        self.loss_cls = loss.get("cls_cross_entropy", {})
        self.loss_reg = loss.get("reg_smooth_l1", {})
        self.loss_dir = loss.get("dir_cross_entropy", {})
        self.loss_seg = loss.get("seg_cross_entropy", {})

        self.device = device
        self.to(device)

    def forward(self, inputs):
        x = inputs.proj_point
        x = torch.cat(x).reshape(len(x), *x[0].shape)
        x = x.permute(0, 3, 1, 2)

        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        cls_pred = self.conv_cls(up1e)
        reg_pred = self.conv_reg(up1e)
        dir_pred = self.conv_dir(up1e)

        #label = inputs.proj_label
        #label = torch.cat(label).reshape(len(label), *label[0].shape)
        
        #cls_gt = torch.full(cls_pred.shape, float(-2**31), device=self.device)
        #for i in range(cls_gt.shape[1]):
        #  tmp = cls_gt[:,i,:,:]
        #  tmp[label == i] = 0.
        #cls_pred = cls_gt
        
        #delta = inputs.proj_delta
        #delta = torch.cat(delta).reshape(len(delta), *delta[0].shape)
        #delta = delta.permute(0, 3, 1, 2)
        #reg_pred = delta
        
        #target_dir = delta[:, -1]
        #target_dir = limit_period(target_dir, 0, 2 * np.pi)
        #target_dir = (target_dir / np.pi).long() % 2
        #dir_gt = torch.full(dir_pred.shape, float(-2**31), device=self.device)
        #for i in range(dir_gt.shape[1]):
        #  tmp = dir_gt[:,i,:,:]
        #  tmp[target_dir == i] = 0.
        #dir_pred = dir_gt

        #cls_pred = cls_pred.cpu()
        #reg_pred = reg_pred.cpu()
        #dir_pred = dir_pred.cpu()
        return [cls_pred, reg_pred, dir_pred], cls_pred

    def get_optimizer(self, params, cfg):
        optimizer = torch.optim.AdamW(params, **cfg)
        return optimizer, None

    def get_loss(self, od_result, ss_result, input):
        score, bbox, dir = od_result

        label = input.proj_label
        label = torch.cat(label).reshape(len(label), *label[0].shape)
        #label = label.cpu()
        loss_cls = F.cross_entropy(score, label, weight=self.ss_class_weight, ignore_index=-1)

        mask = ((label >= 0) & (label < self.od_num_classes)).unsqueeze(axis=1)

        delta = input.proj_delta
        delta = torch.cat(delta).reshape(len(delta), *delta[0].shape)
        #delta = delta.cpu()
        delta = delta.permute(0, 3, 1, 2)

        r0 = torch.sin(bbox[:, -1:]) * torch.cos(delta[:, -1:])
        r1 = torch.cos(bbox[:, -1:]) * torch.sin(delta[:, -1:])
        bbox_ = torch.cat([bbox[:, :-1], r0], axis=1)
        delta_ = torch.cat([delta[:, :-1], r1], axis=1)
        loss_reg = F.smooth_l1_loss(bbox_, delta_, reduction='none', beta=self.loss_reg.beta).sum(dim=1, keepdim=True)
        loss_reg = loss_reg[mask].mean()

        target_dir = delta[:, -1]
        target_dir = limit_period(target_dir, 0, 2 * np.pi)
        target_dir = (target_dir / np.pi).long() % 2
        loss_dir = F.cross_entropy(dir, target_dir, reduction='none').unsqueeze(axis=1)
        loss_dir = loss_dir[mask].mean()

        if mask.sum() == 0:
            return {
                'loss_cls': loss_cls,
                'loss_reg': torch.zeros(1).to(self.device),
                'loss_dir': torch.zeros(1).to(self.device),
                'loss_seg': torch.zeros(1).to(self.device)
            }
        else:
            return {
                'loss_cls': loss_cls,
                'loss_reg': loss_reg,
                'loss_dir': loss_dir,
                'loss_seg': torch.zeros(1).to(self.device)
            }

    def preprocess(self, data, attr):
        #print(attr)

        # open a semantic laserscan
        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.do_transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if attr['split'] not in ['test', 'testing']:
          scan = SemLaserScan(sem_color_dict=None,
                              project=True,
                              H=self.sensor_img_H,
                              W=self.sensor_img_W,
                              fov_up=self.sensor_fov_up,
                              fov_down=self.sensor_fov_down,
                              DA=DA,
                              flip_sign=flip_sign,
                              drop_points=drop_points)
        else:
          scan = LaserScan(project=True,
                           H=self.sensor_img_H,
                           W=self.sensor_img_W,
                           fov_up=self.sensor_fov_up,
                           fov_down=self.sensor_fov_down,
                           DA=DA,
                           rot=rot,
                           flip_sign=flip_sign,
                           drop_points=drop_points)

        point = np.array(data['point'][:, 0:3], dtype=np.float32)
        remission = np.array(data['point'][:, 3], dtype=np.float32)
        scan.set_points(point, remission)
        proj_point = np.concatenate((scan.proj_xyz, scan.proj_range[..., np.newaxis], scan.proj_remission[..., np.newaxis]), axis=-1)

        new_data = {'calib': data['calib'], 'point': point, 'proj_point': proj_point}

        if attr['split'] not in ['test', 'testing']:
            bounding_boxes = []
            for box in data['bounding_boxes']:
                bounding_boxes.append(box)
            new_data['bbox_objs'] = bounding_boxes
            label = np.array(data['label'], dtype=np.int64) - 1

            delta_xyz = np.zeros((point.shape[0], 1, 3), dtype=np.float32)
            delta_wlhr = np.zeros((point.shape[0], 1, 4), dtype=np.float32)
            mask_all = (label >= self.od_num_classes)
            for bb in bounding_boxes:
                lbl = self.od_name2lbl.get(bb.label_class, self.od_num_classes)
                if lbl >= self.od_num_classes: continue
                x_axis = bb.left
                y_axis = bb.front
                z_axis = bb.up
                bbox = bb.to_xyzwlhr()
                x,y,z,w,l,h,r = bbox
                R = np.matrix([x_axis, y_axis, z_axis]).T
                origin = bbox[0:3] + np.array([-w/2, -l/2, 0]) @ R.I
                local = (point - origin) @ R  
                mask0 = np.logical_and(local[:,0] >= 0, local[:,0] <= w)
                mask1 = np.logical_and(local[:,1] >= 0, local[:,1] <= l)
                mask2 = np.logical_and(local[:,2] >= 0, local[:,2] <= h)
                mask3 = (label == lbl)
                mask = mask0 & mask1 & mask2 & mask3
                #print(lbl, mask.sum(), bbox)
                mask_all |= mask
                diagonal = np.sqrt(w**2 + l**2)
                delta_x = (x - point[:,0]) / diagonal
                delta_y = (y - point[:,1]) / diagonal
                delta_z = (z - point[:,2]) / h
                anchor_size = self.anchor_sizes[lbl]
                delta_w = np.log(w / anchor_size[0])
                delta_l = np.log(l / anchor_size[1])
                delta_h = np.log(h / anchor_size[2])
                delta_r = r + np.pi / 2
                delta_xyz[mask] = np.stack([delta_x, delta_y, delta_z], axis=1).reshape(delta_xyz.shape)[mask]
                delta_wlhr[mask] = delta_w, delta_l, delta_h, delta_r
                break

            delta = np.concatenate([delta_xyz, delta_wlhr], axis=-1).squeeze()
            proj_delta = np.zeros((*proj_point.shape[0:2], 7), dtype=np.float32)
            proj_delta[scan.proj_y, scan.proj_x] = delta[scan.order]
            new_data['proj_delta'] = proj_delta
            label[mask_all == 0] = -1 
            new_data['label'] = label
            proj_label = np.full((*proj_point.shape[0:2], 1), -1,
                                        dtype=np.int64)
            proj_label[scan.proj_y, scan.proj_x] = label[scan.order]
            new_data['proj_label'] = proj_label.squeeze()
            #import cv2
            #import sys
            #cv2.imwrite('proj_range.png', scan.proj_range)
            #arr = np.zeros((*scan.proj_sem_label.shape, 3), dtype=np.uint8)
            #arr[proj_label==0] = [  0,  0,230] #pedestrian
            #arr[proj_label==1] = [220, 20, 60] #bike
            #arr[proj_label==2] = [255,158,  0] #car
            #arr[proj_label==3] = [255, 99, 71] #other_vehicle
            #arr[proj_label==4] = [  0,207,191] #driveable_surface
            #arr[proj_label==5] = [ 75,  0, 75] #sidewalk
            #arr[proj_label==6] = [112,180, 60] #terrain
            #arr[proj_label==7] = [222,184,135] #manmade
            #arr[proj_label==8] = [  0,175,  0] #vegetation
            #cv2.imwrite('proj_label.png', arr)
            #np.set_printoptions(threshold=np.inf)
            #mask = (proj_label >= 0) & (proj_label < self.od_num_classes)
            #print(proj_delta[mask.squeeze(),])
            #sys.exit()

        return new_data

    def transform(self, data, attr):
        return data

    def inference_end(self, result, input):
        od_result, ss_result = result

        score_b, bbox_b, dir_b = od_result
        proj_point = input.proj_point
        point_b = torch.cat(proj_point).reshape(len(proj_point), *proj_point[0].shape)
        #point_b = point_b.cpu()

        od_pred = []
        for _calib, _point, _bbox, _score, _dir in zip(
            input.calib, point_b, bbox_b, score_b, dir_b):
            point = _point
            bbox = _bbox
            score = _score[0:self.od_num_classes,:,:]
            dir = _dir

            cls = torch.argmax(score, dim=0)
            p_w = torch.exp(bbox[3,:,:])
            p_l = torch.exp(bbox[4,:,:])
            p_h = torch.exp(bbox[5,:,:])
            p_r = bbox[6,:,:]
            for i in range(len(self.anchor_sizes)):
              anchor_size = self.anchor_sizes[i]
              mask = (cls == i)
              p_w[mask] *= anchor_size[0]
              p_l[mask] *= anchor_size[1]
              p_h[mask] *= anchor_size[2]
              p_r[mask] -= np.pi / 2
            diagonal = torch.sqrt(p_w**2 + p_l**2)
            p_x = point[:,:,0] + bbox[0,:,:] * diagonal
            p_y = point[:,:,1] + bbox[1,:,:] * diagonal
            p_z = point[:,:,2] + bbox[2,:,:] * p_h
            
            bbox = torch.stack([p_x, p_y, p_z, p_w, p_h, p_l, p_r], dim=-1)
            bbox = bbox.reshape(-1, 7)
            score = score.permute(1, 2, 0).reshape(-1, self.od_num_classes)
            score = F.softmax(score, dim=-1)
            dir = dir.permute(1, 2, 0).reshape(-1, 2)
            dir = torch.argmax(dir, dim=-1)

            if score.shape[0] > self.nms_pre:
                max_score, _ = score.max(dim=-1)
                _, topk_ind = max_score.topk(self.nms_pre)
                bbox = bbox[topk_ind, :]
                #print(bbox)
                score = score[topk_ind, :]
                dir = dir[topk_ind]

            idx = multiclass_nms(bbox, score, self.score_thr)

            lbl = [
                torch.full((len(idx[i]),), i, dtype=torch.long)
                for i in range(self.od_num_classes)
            ]
            lbl = torch.cat(lbl)

            score = [score[idx[i], i] for i in range(self.od_num_classes)]
            score = torch.cat(score)

            idx = torch.cat(idx)
            bbox = bbox[idx]
            dir = dir[idx]

            if bbox.shape[0] > 0:
                dir_rot = limit_period(bbox[..., 6] - self.dir_offset, 1, np.pi)
                bbox[..., 6] = (dir_rot + self.dir_offset +
                                  np.pi * dir.to(bbox.dtype))

            bbox = bbox.cpu().detach().numpy()
            score = score.cpu().detach().numpy()
            dir = dir.cpu().detach().numpy()
            lbl = lbl.cpu().detach().numpy()
            od_pred.append([])

            world_cam, cam_img = None, None
            if _calib is not None:
                world_cam = _calib.get('world_cam', None)
                cam_img = _calib.get('cam_img', None)

            for bb, sc, lb in zip(bbox, score, lbl):
                dim = bb[3:6]
                pos = bb[0:3] + [0, 0, dim[1]/2]
                yaw = bb[6]
                name = self.od_lbl2name.get(lb, "ignore")
                od_pred[-1].append(
                    BEVBox3D(pos, dim, yaw, name, sc, world_cam, cam_img))

        return od_pred, None


MODEL._register_module(mtSalsaNext, 'torch')
