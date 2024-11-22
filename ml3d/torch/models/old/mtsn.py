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
from ..modules.losses.semseg_loss import WeightedCrossEntropyLoss
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

class SalsaNext(nn.Module):

    def __init__(self, num_classes):
        super(SalsaNext, self).__init__()

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

        self.conv_cls = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
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

        #label = inputs.label_t
        #label = torch.cat(label).reshape(len(label), *label[0].shape)
        
        #cls_gt = torch.full(cls_pred.shape, float(-2**31))
        #for i in range(cls_gt.shape[1]):
        #  tmp = cls_gt[:,i,:,:]
        #  tmp[label == i] = 0.
        #cls_pred = cls_gt
        
        #cls_pred = cls_pred.cpu()
        return cls_pred

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
                 #layer_nums=[3, 5, 5],
                 layer_nums=[1, 1, 1],
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
        #dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

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

class mtSalsaNext(BaseModel):

    def __init__(self,
                 name="mtSalsaNext",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 voxel_size=[0.25, 0.25, 8],

                 od_classes=['pedestrian', 'bike', 'car', 'other_vehicle'],
                 od_num_classes=4,
                 ss_classes=['pedestrian', 'bike', 'car', 'other_vehicle', 'driveable_surface', 'sidewalk', 'terrain', 'manmade', 'vegetation'],
                 ss_num_classes=9,
                 ignore_label=-1,
                 num_per_class=[46687, 7834, 501416, 410507, 4069879, 746905, 696526, 2067585, 1565272],
                 sensor_fov_up=10,
                 sensor_fov_down=-30,
                 sensor_img_width=2048,
                 sensor_img_height=64,
                 sensor_img_means=[10.,  0.,  0., -1., 0.2],
                 sensor_img_stds =[10., 10., 10.,  1., 0.2],
                 max_points=150000,
                 backbone={},
                 neck={},
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
        self.ignore_label = ignore_label
        num_per_class = np.array(num_per_class, dtype=np.float32)
        r = num_per_class / float(sum(num_per_class))
        self.ss_class_weight = torch.tensor(1/(r+0.02)).to(device)

        self.sensor_img_H = sensor_img_height
        self.sensor_img_W = sensor_img_width
        self.sensor_img_means = torch.tensor(sensor_img_means,
                                             dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor_img_stds,
                                            dtype=torch.float)
        self.sensor_fov_up = sensor_fov_up
        self.sensor_fov_down = sensor_fov_down
        self.max_points = max_points

        self.min_x = point_cloud_range[0]
        self.max_x = point_cloud_range[3]
        self.min_y = point_cloud_range[1]
        self.max_y = point_cloud_range[4]
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.bev_w = int((self.max_x - self.min_x) / self.voxel_size[0])
        self.bev_h = int((self.max_y - self.min_y) / self.voxel_size[1])

        self.do_transform = False
        #self.augmenter = ObjdetAugmentation(self.cfg.augment, seed=self.rng)

        self.semseg = SalsaNext(num_classes=self.ss_num_classes)
        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(num_classes=self.od_num_classes, **head)

        self.loss_cls = FocalLoss(**loss.get("cls_focal", {}))
        self.loss_reg = SmoothL1Loss(**loss.get("reg_smooth_l1", {}))
        self.loss_dir = CrossEntropyLoss(**loss.get("dir_cross_entropy", {}))
        self.loss_seg = WeightedCrossEntropyLoss(**loss.get("seg_cross_entropy", {}))

        self.device = device
        self.to(device)

    def forward(self, inputs):
        x = inputs.point_t
        x = torch.cat(x).reshape(len(x), *x[0].shape)
        x = x.permute(0, 3, 1, 2)

        ss_result = self.semseg(x)

        pnt = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        idx = (pnt[:,0] - self.min_x) / self.voxel_size[0]
        idx = torch.clamp(idx.to(torch.long), 0, self.bev_w - 1)
        idy = (pnt[:,1] - self.min_y) / self.voxel_size[1]
        idy = torch.clamp(idy.to(torch.long), 0, self.bev_h - 1)
        cls = torch.argmax(ss_result, dim=1)
        cls = cls.view(cls.shape[0], cls.shape[1] * cls.shape[2])
        ind = ((idy * self.bev_w + idx) * self.od_num_classes + cls).cpu()
        msk = ((pnt[:,3] >= 0.) & (cls < self.od_num_classes)).cpu()
        pnt = pnt[:,2].cpu()
        bev = torch.zeros(
              (pnt.shape[0], self.bev_h * self.bev_w * self.od_num_classes, 3),
              dtype=torch.float32, device='cpu')
        for b in range(pnt.shape[0]):
          msk_ = msk[b]
          pnt_ = pnt[b, msk_]
          ind_ = ind[b, msk_]
          bev_ = bev[b]
          for i in range(0, pnt_.shape[0], 2):
            z = pnt_[i]
            a = ind_[i]
            num, min, max = bev_[a]
            #print(b, i, z, a, num, min, max)
            if (num == 0) or (z < min): min = z
            if (num == 0) or (z > max): max = z
            bev_[a] = torch.tensor((num+1, min, max))
 
        bev = bev.view(
              bev.shape[0], self.bev_h, self.bev_w, self.od_num_classes * 3)\
              .permute(0, 3, 1, 2).to(self.device)
        x = self.backbone(bev)
        x = self.neck(x)
        od_result = self.bbox_head(x)

        return od_result, ss_result

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, **cfg_pipeline.scheduler)
        return optimizer, scheduler

    def get_loss(self, od_result, ss_result, input):
        score, bbox, dir = od_result
        gt_cls = input.cls_t
        gt_reg = input.reg_t

        # generate and filter bboxes
        target_bbox, target_idx, pos_idx, neg_idx = self.bbox_head.assign_bboxes(bbox, gt_reg)
        avg_factor = pos_idx.size(0)

        # classification loss
        score = score.permute((0, 2, 3, 1)).reshape(-1, self.od_num_classes)
        target_cls = torch.full((score.size(0),),
                                   self.od_num_classes,
                                   device=score.device,
                                   dtype=gt_cls[0].dtype)
        target_cls[pos_idx] = torch.cat(gt_cls, axis=0)[target_idx]

        all_idx = torch.cat([pos_idx, neg_idx], axis=0)
        loss_cls = self.loss_cls(score[all_idx], target_cls[all_idx], avg_factor=avg_factor)

        # remove invalid labels
        cond = (target_cls[pos_idx] >= 0) & (target_cls[pos_idx] < self.od_num_classes)
        pos_idx = pos_idx[cond]
        target_idx = target_idx[cond]
        target_bbox = target_bbox[cond]

        bbox = bbox.permute(
            (0, 2, 3, 1)).reshape(-1, self.bbox_head.box_code_size)[pos_idx]
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

        label = input.label_t
        label = torch.cat(label).reshape(len(label), *label[0].shape)
        loss_seg = self.loss_seg(ss_result, label, ignore_index=self.ignore_label, weight=self.ss_class_weight)

        return {
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_dir': loss_dir,
            'loss_seg': loss_seg
        }

    @staticmethod
    def in_range_bev(box_range, box):
        return (box[0] > box_range[0]) & (box[1] > box_range[1]) & (
            box[0] < box_range[2]) & (box[1] < box_range[3])

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

        new_data = {'calib': data['calib'], 'point': point, 'point_t': proj_point, 'order': scan.order, 'proj_x': scan.proj_x, 'proj_y': scan.proj_y}

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
                self.od_name2lbl.get(bb.label_class, len(self.od_classes))
                for bb in filtered_boxes
            ], dtype=np.int64)
            new_data['reg_t'] = np.array(
                [bb.to_xyzwlhr() for bb in filtered_boxes], dtype=np.float32)
            label = np.array(data['label'], dtype=np.int64) - 1
            new_data['label'] = label
            proj_label = np.full((*proj_point.shape[0:2], 1), -1, dtype=np.int64)
            proj_label[scan.proj_y, scan.proj_x] = label.reshape([*label.shape, 1])[scan.order]
            new_data['label_t'] = proj_label.squeeze()

        return new_data

    def transform(self, data, attr):
        return data

    def inference_end(self, result, input):
        od_result, ss_result = result
        bbox_b, score_b, label_b = self.bbox_head.get_bboxes(*od_result)

        od_pr = []
        for _calib, _bbox, _score, _label in zip(input.calib, bbox_b,
                                                     score_b, label_b):
            bbox = _bbox.cpu().detach().numpy()
            score = _score.cpu().detach().numpy()
            label = _label.cpu().detach().numpy()
            od_pr.append([])

            world_cam, cam_img = None, None
            if _calib is not None:
                world_cam = _calib.get('world_cam', None)
                cam_img = _calib.get('cam_img', None)

            for bb, sc, la in zip(bbox, score, label):
                dim = bb[[3, 5, 4]]
                pos = bb[:3] + [0, 0, dim[1] / 2]
                yaw = bb[-1]
                name = self.od_lbl2name.get(la, "ignore")
                od_pr[-1].append(
                    BEVBox3D(pos, dim, yaw, name, sc, world_cam, cam_img))

        ss_pr = []
        ss_sc = []
        for pnt, ord, p_x, p_y, res in zip(input.point, input.order, input.proj_x, input.proj_y, ss_result):
            sc = F.softmax(res, dim=0).permute(1, 2, 0).cpu().detach().numpy()
            sa = np.zeros((len(pnt), self.ss_num_classes), dtype=np.float32)
            sa[ord] = sc[p_y,p_x]
            ss_sc.append(sa)
            pr = sc.argmax(axis=2)
            pa = np.zeros(len(pnt), dtype=np.int64)
            pa[ord] = pr[p_y,p_x]
            ss_pr.append(pa)

        return od_pr, ss_pr, ss_sc

MODEL._register_module(mtSalsaNext, 'torch')
