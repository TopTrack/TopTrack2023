from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset

from .dataset.mot import MOT
from .dataset.mot20 import MOT20
from .dataset.kitti_tracking import KITTITracking
from .dataset.crowdhuman import CrowdHuman


dataset_factory = {
  'mot': MOT,
  'mot20': MOT20,
  'kitti_tracking': KITTITracking,
  'crowdhuman': CrowdHuman,
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
