import numpy as np
from scipy.spatial.transform import Rotation as R
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
import logging

from sklearn.neighbors import KDTree
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import DataProcessing
from ..utils import make_dir, DATASET
from .utils import BEVBox3D
import open3d as o3d

log = logging.getLogger(__name__)


class mtNuScenes(BaseDataset):
    """This class is used to create a dataset based on the NuScenes
    dataset, and used in visualizer, training, or testing.

    The dataset is best for semantic scene understanding.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='mtNuScenes',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 test_result_folder='./test',
                 num_channels=5,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (mt3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         test_result_folder=test_result_folder,
                         **kwargs)

        self.train_info = {}
        self.val_info = {}
        self.val1_info = {}
        self.val2_info = {}

        if os.path.exists(join(info_path, 'infos_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'infos_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'infos_val.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val1.pkl')):
            self.val1_info = pickle.load(
                open(join(info_path, 'infos_val1.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val2.pkl')):
            self.val2_info = pickle.load(
                open(join(info_path, 'infos_val2.pkl'), 'rb'))

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

        data_config = join(dirname(abspath(__file__)), '_resources/',
                           'mt-nuscenes.yaml')
        DATA = yaml.safe_load(open(data_config, 'r'))

        # make lookup table for mapping
        remap_dict = DATA["learning_map_inv"]
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.uint8)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.uint8)
        remap_lut_val[list(remap_dict_val.keys())] = list(
            remap_dict_val.values())

        self.remap_lut = remap_lut
        self.remap_lut_val = remap_lut_val

        self.num_channels = num_channels

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
           0: 'unlabeled',
           1: 'pedestrian',
           2: 'bike',
           3: 'car',
           4: 'other_vehicle',
           5: 'driveable_surface',
           6: 'sidewalk',
           7: 'terrain',
           8: 'manmade',
           9: 'vegetation'
        }
        return label_to_names

    def read_lidar(self, path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, self.num_channels)

    @staticmethod
    def read_semseg(path):
        """Reads semseg data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.uint8)

    @staticmethod
    def read_label(info, calib):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        mask = info['num_lidar_pts'] != 0
        boxes = info['gt_boxes'][mask]
        names = info['gt_names'][mask]

        objects = []
        for name, box in zip(names, boxes):

            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])

            yaw = ry - np.pi
            yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

            world_cam = calib['world_cam']

            objects.append(BEVBox3D(center, size, yaw, name, -1.0, world_cam))
            objects[-1].yaw = ry

        return objects

    def read_cams(self, cam_dict):
        """Reads image data from the cam dict provided.

        Args:
            cam_dict (Dict): Mapping from camera names to dict with image
                information ('data_path', 'sensor2lidar_translation',
                'sensor2lidar_rotation', 'cam_intrinsic').

        Returns:
            A dict with keys as camera names and value as images.
        """
        assert [Path(val['data_path']).exists() for _, val in cam_dict.items()]

        res_dict = dict()
        for cam in cam_dict.keys():
            res_dict[cam] = dict()
            res_dict[cam]['img'] = np.array(
                o3d.io.read_image(cam_dict[cam]['data_path']))

            # obtain lidar to cam transformation matrix
            lidar2cam_r = np.linalg.inv(cam_dict[cam]['sensor2lidar_rotation'])
            lidar2cam_t = cam_dict[cam][
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_dict[cam]['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            # obtain lidar to image transformation matrix
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            res_dict[cam]['lidar2cam_rt'] = lidar2cam_rt
            res_dict[cam]['lidar2img_rt'] = lidar2img_rt
            res_dict[cam]['cam_intrinsic'] = cam_dict[cam]['cam_intrinsic']

        return res_dict

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'train', 'val', 'val1', 'val2' or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return mtNuScenesSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'train', 'val', 'val1', 'val2', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
                split name should be one of 'train', 'val', 'val1', 'val2' or 'all'.
        """
        if split in ['train']:
            return self.train_info
        elif split in ['val']:
            return self.val_info
        elif split in ['val1']:
            return self.val1_info
        elif split in ['val2']:
            return self.val2_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, data, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            data: The attributes that correspond to the outputs passed in
            results.
        """
        make_dir(self.cfg.test_result_folder)

        od_results, ss_results = results
        for od, ss, da, at in zip(od_results, ss_results, data, attr):
            name = at['name']

            point = da['point']
            path = join(self.cfg.test_result_folder, name + '.bin')
            point.tofile(path)

            #path = join(self.cfg.test_result_folder, name + '.txt')
            #f = open(path, 'w')
            #for box in od:
            #    f.write(box.to_kitti_format(box.confidence))
            #    f.write('\n')

            pred = ss + 1
            pred = self.remap_lut[pred].astype(np.uint8)
            path = join(self.cfg.test_result_folder, name + '.label')
            pred.tofile(path)


class mtNuScenesSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)
        self.path_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))

        self.split = split
        self.dataset = dataset
        self.remap_lut_val = dataset.remap_lut_val

        super().__init__(dataset, split=split)

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']

        world_cam = np.eye(4)
        world_cam[:3, :3] = R.from_quat(info['lidar2ego_rot']).as_matrix()
        world_cam[:3, -1] = info['lidar2ego_tr']
        calib = {'world_cam': world_cam.T}

        pc = self.dataset.read_lidar(lidar_path)
        boxes = self.dataset.read_label(info, calib)
        semseg_path = info['semseg_path']
        label = self.dataset.read_semseg(semseg_path)
        label = self.remap_lut_val[label]

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': boxes,
            'label': label,
        }

        if 'cams' in info:
            data['cams'] = self.dataset.read_cams(info['cams'])

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        #print(info)
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr

DATASET._register_module(mtNuScenes)