import logging
import re
import numpy as np
import torch
import time

from datetime import datetime
from os.path import exists, join
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader, ConcatBatcher
from torch.utils.tensorboard import SummaryWriter
# pylint: disable-next=unused-import
from open3d.visualization.tensorboard_plugin import summary
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets.utils import BEVBox3D

from ...metrics.mAP import mAP
from ..modules.losses import filter_valid_label

log = logging.getLogger(__name__)

import subprocess
import json

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]


class ObjectDetection(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self,
                 model,
                 dataset=None,
                 name='ObjectDetection',
                 main_log_dir='./logs/',
                 device='cuda',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)
        self.split = split

    def run_inference(self, data):
        """Run inference on given data.

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        model = self.model

        model.eval()

        # If run_inference is called on raw data.
        if isinstance(data, dict):
            batcher = ConcatBatcher(self.device, model.cfg.name)
            data = batcher.collate_fn([{
                'data': data,
                'attr': {
                    'split': 'test'
                }
            }])

        data.to(self.device)

        with torch.no_grad():
            results = model(data)
            bbox = model.inference_end(results, data)

        return bbox

    @staticmethod
    def worker_init_fn(x):
        return np.random.seed(
                   x + np.uint32(torch.utils.data.get_worker_info().seed))
      
    def run_test(self):
        """Run test with test data split, computes mean average precision of the
        prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        self.load_ckpt(model.cfg.ckpt_path)

        model.eval()

        test_split = TorchDataloader(dataset=dataset.get_split('test'),
                                     preprocess=model.preprocess,
                                     transform=None,
                                     use_cache=False,
                                     shuffle=False)

        log.info("Started testing")
        for idx in tqdm(range(len(test_dataset)), desc='testing'):
            data = test_split[idx]['data']
            attr = test_split[idx]['attr']
            results = model.run_inference(data)
            dataset.save_test_result(results, [data], [attr])

    def run_valid(self, epoch=0):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.

        Args:
            epoch (int): step for TensorBoard summary. Defaults to 0 if
                unspecified.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.eval()

        #timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        #
        #log.info("DEVICE : {}".format(device))
        #log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        #log.info("Logging in file : {}".format(log_file_path))
        #log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        if self.split == 'train':
          valid_dataset = dataset.get_split('val1')
        else:
          valid_dataset = dataset.get_split(self.split)
          self.load_ckpt(model.cfg.ckpt_path, is_resume=True)

        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      shuffle=True,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_valid', None))

        valid_loader = DataLoader(valid_split,
                                  batch_size=cfg.val_batch_size,
                                  num_workers=cfg.get('num_workers', 4),
                                  pin_memory=cfg.get('pin_memory', False),
                                  collate_fn=batcher.collate_fn,
                                  sampler=None)

        log.info("Started validation")

        self.valid_losses = {}

        pr = []
        gt = []
        total_time = 0.
        max_usage = 0
        with torch.no_grad():
            #iter = 0
            for data in tqdm(valid_loader, desc='validation'):
                #if iter > 10: break
                #iter += 1
                data.to(device)
                start_time = time.time()
                result = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                mem_usage = int(get_gpu_info()[0]['memory.used'])
                if mem_usage > max_usage: max_usage = mem_usage
                #if torch.isnan(torch.cat(result, dim=1)).any():
                #  print("NaN occured at model calculation!")
                #  continue

                loss = model.get_loss(result, data)
                if torch.isnan(torch.Tensor(list(loss.values()))).any():
                  print("NaN occured at loss calculation!")
                  continue

                for l, v in loss.items():
                    if l not in self.valid_losses:
                        self.valid_losses[l] = []
                    self.valid_losses[l].append(float(v.cpu().numpy()))

                # convert to bboxes for mAP evaluation
                bbox = model.inference_end(result, data)

                pr.extend([BEVBox3D.to_dicts(b) for b in bbox])
                gt.extend([BEVBox3D.to_dicts(b) for b in data.bbox])

        log.info("Total inference time: {:.3f}s".format(total_time))
        log.info("Maximum memory usage: {:d}MiB".format(max_usage))

        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)
        desc += " > loss: %.03f" % sum_loss
        log.info(desc)

        overlaps = cfg.get("overlaps", [0.5])
        similar_classes = cfg.get("similar_classes", {})
        difficulties = cfg.get("difficulties", [0])
        ap = mAP(pr,
                 gt,
                 model.classes,
                 difficulties,
                 overlaps,
                 similar_classes=similar_classes)
        log.info("")
        log.info("=============== mAP BEV ===============")
        log.info(("class \\ difficulty  " +
                  "{:>5} " * len(difficulties)).format(*difficulties))
        for i, c in enumerate(model.classes):
            log.info(("{:<20} " + "{:>5.2f} " * len(difficulties)).format(
                c + ":", *ap[i, :, 0]))
        mAP_BEV = np.mean(ap[:, -1])
        log.info("Overall: {:.2f}".format(mAP_BEV))
        self.valid_losses["mAP BEV"] = mAP_BEV
        ap = mAP(pr,
                 gt,
                 model.classes,
                 difficulties,
                 overlaps,
                 similar_classes=similar_classes,
                 bev=False)
        log.info("")
        log.info("=============== mAP  3D ===============")
        log.info(("class \\ difficulty  " +
                  "{:>5} " * len(difficulties)).format(*difficulties))
        for i, c in enumerate(model.classes):
            log.info(("{:<20} " + "{:>5.2f} " * len(difficulties)).format(
                c + ":", *ap[i, :, 0]))
        mAP_3D = np.mean(ap[:, -1])
        log.info("Overall: {:.2f}".format(mAP_3D))
        self.valid_losses["mAP 3D"] = mAP_3D
        return mAP_BEV, mAP_3D

    def run_train(self):
        """Run training with train data split."""
        torch.manual_seed(self.rng.integers(np.iinfo(
            np.int32).max))  # Random reproducible seed for torch
        model = self.model
        device = self.device
        dataset = self.dataset

        cfg = self.cfg

        log.info("DEVICE : {}".format(device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir,
                             'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        train_dataset = dataset.get_split('train')
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      shuffle=True,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))

        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            sampler=None,
            worker_init_fn=self.worker_init_fn
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        weight_cls = float(model.loss_cls.loss_weight)
        weight_reg = float(model.loss_reg.loss_weight)
        weight_dir = float(model.loss_dir.loss_weight)
        log_cls = torch.nn.Parameter(float(model.loss_cls.log_weight) * torch.ones(1))
        log_reg = torch.nn.Parameter(float(model.loss_reg.log_weight) * torch.ones(1))
        log_dir = torch.nn.Parameter(float(model.loss_dir.log_weight) * torch.ones(1))
        if torch.cuda.is_available():
          log_cls = torch.nn.Parameter(log_cls.cuda())
          log_reg = torch.nn.Parameter(log_reg.cuda())
          log_dir = torch.nn.Parameter(log_dir.cuda())
        log_weight_params = dict(
          log_cls=log_cls,
          log_reg=log_reg,
          log_dir=log_dir,
        )
        params = [{'params': model.parameters()},
                  {'params': {**log_weight_params}.values()}]

        self.optimizer, self.scheduler = model.get_optimizer(cfg)

        is_resume = model.cfg.get('is_resume', False)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.tensorboard_dir))

        record_summary = 'train' in cfg.get('summary').get('record_for', [])

        log.info("Started training")

        best_BEV = 0.
        best_3D = 0.
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')

            model.train()
            self.losses = {}

            #iter = 0
            process_bar = tqdm(train_loader, desc='training')
            for data in process_bar:
                #if iter > 10: break
                #iter += 1
                data.to(device)
                result = model(data)
                #if torch.isnan(torch.cat(result, dim=1)).any():
                #  print("NaN occured at model calculation!")
                #  continue

                loss = model.get_loss(result, data)
                if torch.isnan(torch.Tensor(list(loss.values()))).any():
                  print("NaN occured at loss calculation!")
                  continue
                  
                if weight_cls >= 0.:
                  loss_cls = loss['loss_cls'] * weight_cls
                else:
                  loss_cls = loss['loss_cls'] * torch.exp(-log_cls)
                  loss_cls += log_cls if log_cls > 0. else 0.
                if weight_reg >= 0.:
                  loss_reg = loss['loss_reg'] * weight_reg
                else:
                  loss_reg = loss['loss_reg'] * torch.exp(-log_reg)
                  loss_reg += log_reg if log_reg > 0. else 0.
                if weight_dir >= 0.:
                  loss_dir = loss['loss_dir'] * weight_dir
                else:
                  loss_dir = loss['loss_dir'] * torch.exp(-log_dir)
                  loss_dir += log_dir if log_dir > 0. else 0.
                loss_sum = loss_cls + loss_reg + loss_dir

                self.optimizer.zero_grad()
                loss_sum.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(
                        model.parameters(), model.cfg.grad_clip_norm)

                self.optimizer.step()

                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(float(v.cpu().detach().numpy()))

            if self.scheduler is not None:
                self.scheduler.step()

            # --------------------- validation
            if epoch % cfg.get("validation_freq", 1) == 0:
                mAP_BEV, mAP_3D = self.run_valid()
                if mAP_BEV > best_BEV:
                  best_BEV = mAP_BEV
                  self.save_best(epoch, 'BEV')
                if mAP_3D > best_3D:
                  best_3D = mAP_3D
                  self.save_best(epoch, '3D')

            self.save_logs(writer, epoch)
            #if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
            #    self.save_ckpt(epoch)

    def save_logs(self, writer, epoch):
        for key, val in self.losses.items():
            writer.add_scalar("train/" + key, np.mean(val), epoch)
        if (epoch % self.cfg.get("validation_freq", 1)) == 0:
            for key, val in self.valid_losses.items():
                writer.add_scalar("valid/" + key, np.mean(val), epoch)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        epoch = 0
        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
                epoch = int(re.findall(r'\d+', ckpt_path)[-1]) + 1
            else:
                log.info('Initializing from scratch.')
                return epoch

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info('Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info('Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
            # scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def save_best(self, epoch, desc):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
            # scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'best_{desc:s}.pth'))
        log.info(f'Epoch {epoch:3d}: save best {desc:s} to {path_ckpt:s}')

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary."""
        if hasattr(self, 'cfg_tb'):
            writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
            writer.add_text("Description/Command line", self.cfg_tb['cmd_line'],
                            0)
            writer.add_text('Configuration/Dataset',
                            code2md(self.cfg_tb['dataset'], language='json'), 0)
            writer.add_text('Configuration/Model',
                            code2md(self.cfg_tb['model'], language='json'), 0)
            writer.add_text('Configuration/Pipeline',
                            code2md(self.cfg_tb['pipeline'], language='json'),
                            0)


PIPELINE._register_module(ObjectDetection, "torch")
