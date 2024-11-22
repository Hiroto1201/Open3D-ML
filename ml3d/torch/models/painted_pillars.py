#***************************************************************************************/
#
#    Based on MMDetection3D Library (Apache 2.0 license):
#    https://github.com/open-mmlab/mmdetection3d
#
#    Copyright 2018-2019 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#***************************************************************************************/

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
from ...datasets.augment import ObjdetAugmentation

class PaintedPillars(BaseModel):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "PointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self,
                 name="PaintedPillars",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['pedestrian', 'bike', 'car', 'other_vehicle'],
                 num_classes=4,
                 voxelize={},
                 voxel_encoder={},
                 scatter={},
                 backbone={},
                 neck={},
                 head={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        self.classes = classes
        self.num_classes = num_classes
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.lbl2name = {i: n for i, n in enumerate(classes)}

        self.min_x = point_cloud_range[0]
        self.max_x = point_cloud_range[3]
        self.min_y = point_cloud_range[1]
        self.max_y = point_cloud_range[4]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxelize.voxel_size
        self.grid_x = int((self.max_x - self.min_x) / self.voxel_size[0])
        self.grid_y = int((self.max_y - self.min_y) / self.voxel_size[1])
        self.range_x = self.max_x - self.min_x
        self.range_y = self.max_y - self.min_y

        #self.augmenter = ObjdetAugmentation(self.cfg.augment, seed=self.rng)
        self.voxel_layer = PointPillarsVoxelization(
            point_cloud_range=point_cloud_range, **voxelize)
        self.voxel_encoder = StatisticalPillarProcessing(
            point_cloud_range=point_cloud_range, **voxel_encoder)

        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.head_det = Anchor3DHead(num_classes=self.num_classes, **head)

        self.loss_cls = FocalLoss(**loss.get("cls_focal", {}))
        self.loss_reg = SmoothL1Loss(**loss.get("reg_smooth_l1", {}))
        self.loss_dir = CrossEntropyLoss(**loss.get("dir_cross_entropy", {}))

        self.device = device
        self.to(device)

    @torch.no_grad()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feats(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        batch_size = coors[-1, 0].item() + 1
        x = self.voxel_encoder(voxels, num_points, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    def forward(self, inputs):
        inputs = inputs.point_t
        x = self.extract_feats(inputs)
        det = self.head_det(x)
        return det

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **cfg_pipeline.scheduler)
        return optimizer, scheduler

    def get_loss(self, result, input):
        score, bbox, dir = result
        gt_cls = input.cls_t
        gt_reg = input.reg_t

        # generate and filter bboxes
        target_bbox, target_idx, pos_idx, neg_idx = self.head_det.assign_bboxes(bbox, gt_reg)
        avg_factor = pos_idx.size(0)

        # classification loss
        score = score.permute((0, 2, 3, 1)).reshape(-1, self.num_classes)
        target_cls = torch.full((score.size(0),),
                                   self.num_classes,
                                   device=score.device,
                                   dtype=gt_cls[0].dtype)
        target_cls[pos_idx] = torch.cat(gt_cls, axis=0)[target_idx]

        all_idx = torch.cat([pos_idx, neg_idx], axis=0)
        loss_cls = self.loss_cls(score[all_idx], target_cls[all_idx], avg_factor=avg_factor)

        # remove invalid labels
        cond = (target_cls[pos_idx] >= 0) & (target_cls[pos_idx] < self.num_classes)
        pos_idx = pos_idx[cond]
        target_idx = target_idx[cond]
        target_bbox = target_bbox[cond]

        bbox = bbox.permute(
            (0, 2, 3, 1)).reshape(-1, self.head_det.box_code_size)[pos_idx]
        dir = dir.permute((0, 2, 3, 1)).reshape(-1, 2)[pos_idx]

        if len(pos_idx) > 0:
            # direction classification loss
            # to discrete bins
            target_dir = torch.cat(gt_reg, axis=0)[target_idx][:, -1]
            target_dir = limit_period(target_dir, 0, 2 * np.pi)
            target_dir = (target_dir / np.pi).long() % 2

            loss_dir = self.loss_dir(dir, target_dir, avg_factor=avg_factor)

            # bbox loss
            # sinus difference transformation
            r0 = torch.sin(bbox[:, -1:]) * torch.cos(target_bbox[:, -1:])
            r1 = torch.cos(bbox[:, -1:]) * torch.sin(target_bbox[:, -1:])

            bbox = torch.cat([bbox[:, :-1], r0], axis=-1)
            target_bbox = torch.cat([target_bbox[:, :-1], r1], axis=-1)

            loss_reg = self.loss_reg(bbox, target_bbox, avg_factor=avg_factor)
        else:
            loss_cls = loss_cls.sum()
            loss_reg = bbox.sum()
            loss_dir = dir.sum()

        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_dir': loss_dir,
        }

    @staticmethod
    def in_range_bev(box_range, box):
        return (box[0] > box_range[0]) & (box[1] > box_range[1]) & (
            box[0] < box_range[2]) & (box[1] < box_range[3])

    def preprocess(self, data, attr):
        point = np.array(data['point'][:, 0:self.voxel_encoder.in_channels], dtype=np.float32)

        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])

        mask = np.where(np.all(np.logical_and(point[:, :3] >= min_val,
                                              point[:, :3] < max_val),
                               axis=-1))
        point = point[mask]
        new_data = {'calib': data['calib'], 'point': point, 'point_t': point}

        if attr['split'] not in ['test', 'testing']:
            """Filter Objects in the given range."""
            pcd_range = np.array(self.point_cloud_range)
            bev_range = pcd_range[[0, 1, 3, 4]]
            filtered_boxes = []
            for box in data['bounding_boxes']:
                if self.in_range_bev(bev_range, box.to_xyzwlhr()):
                    filtered_boxes.append(box)
            new_data['bbox'] = filtered_boxes
            new_data['cls_t'] = np.array([
                self.name2lbl.get(bb.label_class, len(self.classes))
                for bb in filtered_boxes
            ], dtype=np.int64)
            new_data['reg_t'] = np.array(
                [bb.to_xyzwlhr() for bb in filtered_boxes], dtype=np.float32)

        return new_data

    def transform(self, data, attr):
        return data

    def inference_end(self, result, input):
        bbox_b, score_b, label_b = self.head_det.get_bboxes(*result)

        pred = []
        for _calib, _bbox, _score, _label in zip(input.calib, bbox_b,
                                                     score_b, label_b):
            bbox = _bbox.cpu().detach().numpy()
            score = _score.cpu().detach().numpy()
            label = _label.cpu().detach().numpy()
            pred.append([])

            world_cam, cam_img = None, None
            if _calib is not None:
                world_cam = _calib.get('world_cam', None)
                cam_img = _calib.get('cam_img', None)

            for bb, sc, la in zip(bbox, score, label):
                dim = bb[[3, 5, 4]]
                pos = bb[:3] + [0, 0, dim[1] / 2]
                yaw = bb[-1]
                name = self.lbl2name.get(la, "ignore")
                pred[-1].append(
                    BEVBox3D(pos, dim, yaw, name, sc, world_cam, cam_img))

        return pred


MODEL._register_module(PaintedPillars, 'torch')


class PointPillarsVoxelization(torch.nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points=32,
                 max_voxels=[16000, 40000]):
        """Voxelization layer for the PointPillars model.

        Args:
            voxel_size: voxel edge lengths with format [x, y, z].
            point_cloud_range: The valid range of point coordinates as
                [x_min, y_min, z_min, x_max, y_max, z_max].
            max_num_points: The maximum number of points per voxel.
            max_voxels: The maximum number of voxels. May be a tuple with
                values for training and testing.
        """
        super().__init__()
        self.voxel_size = torch.Tensor(voxel_size)
        self.point_cloud_range = point_cloud_range
        self.points_range_min = torch.Tensor(point_cloud_range[:3])
        self.points_range_max = torch.Tensor(point_cloud_range[3:])

        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple) or isinstance(max_voxels, list):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)

    def forward(self, points_feats):
        """Forward function.

        Args:
            points_feats: Tensor with point coordinates and features. The shape
                is [N, 3+C] with N as the number of points and C as the number
                of feature channels.

        Returns:
            (out_voxels, out_coords, out_num_points).
            * out_voxels is a dense list of point coordinates and features for
              each voxel. The shape is [num_voxels, max_num_points, 3+C].
            * out_coords is tensor with the integer voxel coords and shape
              [num_voxels,3]. Note that the order of dims is [z,y,x].
            * out_num_points is a 1D tensor with the number of points for each
              voxel.
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        points = points_feats[:, :3]

        num_voxels = ((self.points_range_max - self.points_range_min) /
                      self.voxel_size).type(torch.int32)
        ans = voxelize(points,
                       torch.LongTensor([0, points.shape[0]]).to(points.device),
                       self.voxel_size, self.points_range_min,
                       self.points_range_max, self.max_num_points, max_voxels)

        # prepend row with zeros which maps to index 0 which maps to void points.
        feats = torch.cat(
            [torch.zeros_like(points_feats[0:1, :]), points_feats])

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = ragged_to_dense(
            ans.voxel_point_indices, ans.voxel_point_row_splits,
            self.max_num_points, torch.tensor(-1)) + 1

        out_voxels = feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[
            1:] - ans.voxel_point_row_splits[:-1]

        # Filter out pillars generated out of bounds of the pseudoimage.
        in_bounds_y = out_coords[:, 1] < num_voxels[1]
        in_bounds_x = out_coords[:, 2] < num_voxels[0]
        in_bounds = torch.logical_and(in_bounds_x, in_bounds_y)

        out_coords = out_coords[in_bounds]
        out_voxels = out_voxels[in_bounds]
        out_num_points = out_num_points[in_bounds]

        return out_voxels, out_coords, out_num_points

class StatisticalPillarProcessing(nn.Module):
    def __init__(self,
                 in_channels=25,
                 max_num_points=20,
                 voxel_size=(0.16, 0.16, 4),
                 point_cloud_range=(0, -40.0, -3, 70.0, 40.0, 1),
                 output_shape=[496, 432],
                 ):
        
        super(StatisticalPillarProcessing, self).__init__()
        
        self.in_channels = in_channels
        self.max_num_points = max_num_points
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.output_shape = output_shape
        self.max_reflect = 255
    
    def forward(self, features, num_points, coors, batch_size):
        #torch.set_printoptions(edgeitems=float('inf'))
        #print("==features==: ", features[0, :])
        #print("==num_points==: ", num_points)
        #print("==coors==: ", coors[0])        
        
        batch_indices = coors[:, 0].long()
        x_indices = coors[:, 3].long()
        y_indices = coors[:, 2].long()

        # 点の数を付与
        feat_num_points = torch.zeros(batch_size, 1, *self.output_shape, device=features.device)
        num_pillars = num_points.size()[0]
        sum_num_points = num_points.sum(dim=0)
        mean_num_points = sum_num_points / num_pillars
        diff_num_points = num_points - mean_num_points
        std_num_points = torch.sqrt((diff_num_points**2).sum(dim=0) / num_pillars)
        norm_num_points = (num_points - mean_num_points) / std_num_points
        feat_num_points[batch_indices, 0, y_indices, x_indices] = norm_num_points
        
        # (反射率/最大値)の平均値を付与
        rel_reflect = features[:, :, 3] / self.max_reflect
        sum_reflect = rel_reflect.sum(dim=1)
        feat_reflect = torch.zeros(batch_size, 1, *self.output_shape, device=features.device)
        feat_reflect[batch_indices, 0, y_indices, x_indices] = sum_reflect / num_points

        # 確率分布を取り出す
        prob = features[:, :, 4:]
        mean_prob = (prob.sum(dim=1).permute(1,0) / num_points).clone().detach()
        feat_prob = torch.zeros(batch_size, self.in_channels-4, *self.output_shape, device=features.device)
        feat_prob[batch_indices, :, y_indices, x_indices] = mean_prob.permute(1,0)
        
        # xy座標の特徴量
        coors_x = features[:, :, 0]
        coors_y = features[:, :, 1]
        pillar_center_x = (coors[:, 3].type_as(features).float()) * self.vx + self.x_offset
        pillar_center_y = (coors[:, 2].type_as(features).float()) * self.vy + self.y_offset
        # x, y相対座標の平均値を付与する
        rel_x = (coors_x - pillar_center_x[:, None]) / self.vx
        rel_y = (coors_y - pillar_center_y[:, None]) / self.vy
        mask = torch.arange(self.max_num_points, device=features.device)[None, :] < num_points[:, None]
        mask = mask.type_as(features)
        rel_x = rel_x * mask
        rel_y = rel_y * mask
        mean_x = rel_x.sum(dim=1) / num_points
        mean_y = rel_y.sum(dim=1) / num_points
        feat_mean_xy = torch.zeros(batch_size, 2, *self.output_shape, device=features.device)
        feat_mean_xy[batch_indices, 0, y_indices, x_indices] = mean_x
        feat_mean_xy[batch_indices, 1, y_indices, x_indices] = mean_y

        # x, y座標の標準偏差
        # 2乗の平均
        mean_sq_x = (rel_x**2).sum(dim=1) / num_points
        mean_sq_y = (rel_y**2).sum(dim=1) / num_points
        # 平均の2乗
        sq_mean_x = mean_x**2
        sq_mean_y = mean_y**2
        diff_x = mean_sq_x - sq_mean_x
        diff_y = mean_sq_y - sq_mean_y
        std_x = torch.sqrt(diff_x)
        std_y = torch.sqrt(diff_y)
        std_x[diff_x < 0] = 0
        std_y[diff_y < 0] = 0
        feat_std_xy = torch.zeros(batch_size, 2, *self.output_shape, device=features.device)
        feat_std_xy[batch_indices, 0, y_indices, x_indices] = std_x
        feat_std_xy[batch_indices, 1, y_indices, x_indices] = std_y
        
        # zの平均値を付与する
        sum_z = features[:, :, 2].sum(dim=1)
        mean_z = sum_z / num_points
        feat_mean_z = torch.zeros(batch_size, 1, *self.output_shape, device=features.device)
        feat_mean_z[batch_indices, 0, y_indices, x_indices] = mean_z

        # z座標の標準偏差
        # 2乗の平均
        mean_sq_z = (features[:, :, 2]**2).sum(dim=1) / num_points
        # 平均の2乗
        sq_mean_z = mean_z**2
        diff_z = mean_sq_z - sq_mean_z
        std_z = torch.sqrt(diff_z)
        std_z[diff_z < 0] = 0
        feat_std_z = torch.zeros(batch_size, 1, *self.output_shape, device=features.device)
        feat_std_z[batch_indices, 0, y_indices, x_indices] = std_z
        
        output = torch.cat([
            feat_num_points,
            feat_reflect,
            feat_mean_xy,
            feat_std_xy,
            feat_mean_z,
            feat_std_z,
            feat_prob,
            ], dim=1)
        
        return output 

class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(in_filters[i],
                          out_channels[i],
                          3,
                          bias=False,
                          stride=layer_strides[i],
                          padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(out_channels[i],
                              out_channels[i],
                              3,
                              bias=False,
                              padding=1))
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False):
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(in_channels=in_channels[i],
                                           out_channels=out_channel,
                                           kernel_size=stride,
                                           stride=stride,
                                           bias=False)

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out


class Anchor3DHead(nn.Module):

    def __init__(self,
                 num_classes=1,
                 in_channels=384,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
                 sizes=[[0.6, 1.0, 1.5]],
                 rotations=[0, 1.57],
                 iou_thr=[[0.35, 0.5]]):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.dir_offset = dir_offset
        self.iou_thr = iou_thr

        if len(self.iou_thr) != num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * num_classes
        assert len(self.iou_thr) == num_classes

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(ranges=ranges,
                                                       sizes=sizes,
                                                       rotations=rotations)
        self.num_anchors = self.anchor_generator.num_base_anchors

        # build box coder
        self.bbox_coder = BBoxCoder()
        self.box_code_size = 7

        self.fp16_enabled = False

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                      1)

        self.init_weights()

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """Initialize conv/fc bias value according to giving probablity."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))

        return bias_init

    @staticmethod
    def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = self.bias_init_with_prob(0.01)
        self.normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        self.normal_init(self.conv_reg, std=0.01)

    def forward(self, x):
        """Forward function on a feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)

        return [cls_score, bbox_pred, dir_cls_preds]

    def assign_bboxes(self, pred_bboxes, target_bboxes):
        """Assigns target bboxes to given anchors.

        Args:
            pred_bboxes (torch.Tensor): Bbox predictions (anchors).
            target_bboxes (torch.Tensor): Bbox targets.

        Returns:
            torch.Tensor: Assigned target bboxes for each given anchor.
            torch.Tensor: Flat index of matched targets.
            torch.Tensor: Index of positive matches.
            torch.Tensor: Index of negative matches.
        """
        # compute all anchors
        anchors = [
            self.anchor_generator.grid_anchors(pred_bboxes.shape[-2:],
                                               device=pred_bboxes.device)
            for _ in range(len(target_bboxes))
        ]

        # compute size of anchors for each given class
        anchors_cnt = torch.tensor(anchors[0].shape[:-1]).prod()
        rot_angles = anchors[0].shape[-2]

        # init the tensors for the final result
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):
            """Inject class dimension in the given indices (...

            z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)
            """
            z = idx // rot_angles
            x = idx % rot_angles

            return z * self.num_classes * rot_angles + j * rot_angles + x

        idx_off = 0
        for i in range(len(target_bboxes)):
            for j, (neg_th, pos_th) in enumerate(self.iou_thr):
                anchors_stride = anchors[i][..., j, :, :].reshape(
                    -1, self.box_code_size)

                if target_bboxes[i].shape[0] == 0:
                    assigned_bboxes.append(
                        torch.zeros((0, 7), device=pred_bboxes.device))
                    target_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    pos_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    neg_idxs.append(
                        torch.zeros((0,),
                                    dtype=torch.long,
                                    device=pred_bboxes.device))
                    continue

                # compute a fast approximation of IoU
                overlaps = bbox_overlaps(box3d_to_bev2d(target_bboxes[i]),
                                         box3d_to_bev2d(anchors_stride))

                # for each anchor the gt with max IoU
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                # for each gt the anchor with max IoU
                gt_max_overlaps, _ = overlaps.max(dim=1)

                pos_idx = max_overlaps >= pos_th
                neg_idx = (max_overlaps >= 0) & (max_overlaps < neg_th)

                # low-quality matching
                for k in range(len(target_bboxes[i])):
                    if gt_max_overlaps[k] >= neg_th:
                        pos_idx[overlaps[k, :] == gt_max_overlaps[k]] = True

                # encode bbox for positive matches
                assigned_bboxes.append(
                    self.bbox_coder.encode(
                        anchors_stride[pos_idx],
                        target_bboxes[i][argmax_overlaps[pos_idx]]))
                target_idxs.append(argmax_overlaps[pos_idx] + idx_off)

                # store global indices in list
                pos_idx = flatten_idx(
                    pos_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt
                neg_idx = flatten_idx(
                    neg_idx.nonzero(as_tuple=False).squeeze(-1),
                    j) + i * anchors_cnt
                pos_idxs.append(pos_idx)
                neg_idxs.append(neg_idx)

            # compute offset for index computation
            idx_off += len(target_bboxes[i])

        return (torch.cat(assigned_bboxes,
                          axis=0), torch.cat(target_idxs, axis=0),
                torch.cat(pos_idxs, axis=0), torch.cat(neg_idxs, axis=0))

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds,
                                                  dir_preds):
            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)
            bboxes.append(b)
            scores.append(s)
            labels.append(l)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        assert cls_scores.size()[-2:] == dir_preds.size()[-2:]

        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:],
                                                     device=cls_scores.device)
        anchors = anchors.reshape(-1, self.box_code_size)

        dir_preds = dir_preds.permute(1, 2, 0).reshape(-1, 2)
        dir_scores = torch.max(dir_preds, dim=-1)[1]

        cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_scores.sigmoid()

        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, self.box_code_size)

        if scores.shape[0] > self.nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(self.nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_scores = dir_scores[topk_inds]

        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        idxs = multiclass_nms(bboxes, scores, self.score_thr)

        labels = [
            torch.full((len(idxs[i]),), i, dtype=torch.long)
            for i in range(self.num_classes)
        ]
        labels = torch.cat(labels)

        scores = [scores[idxs[i], i] for i in range(self.num_classes)]
        scores = torch.cat(scores)

        idxs = torch.cat(idxs)
        bboxes = bboxes[idxs]
        dir_scores = dir_scores[idxs]

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, 1, np.pi)
            bboxes[..., 6] = (dir_rot + self.dir_offset +
                              np.pi * dir_scores.to(bboxes.dtype))

        return bboxes, scores, labels
