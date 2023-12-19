from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset, UAVDataset
from .dataset.jde_depth import UAVDataset_Depth

dataset_factory = {
  'jde': JointDataset, 
  'default': JointDataset, 
  'mot': JointDataset, 
  'visdrone': UAVDataset, 
  'uavdt': UAVDataset, 

}

depth_dataset_factory = {
  'visdrone': UAVDataset_Depth, 
  'uavdt': UAVDataset_Depth, 
  'mot17': UAVDataset_Depth, 
  'kitti': UAVDataset_Depth, 
}


def get_dataset(dataset, task):
  if task == 'mot':
    return dataset_factory[dataset]
  elif task == 'mot_depth':
    return depth_dataset_factory[dataset]
  else:
    return None
  
