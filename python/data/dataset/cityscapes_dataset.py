from __future__ import print_function

import numpy as np
import os
import glob

from .common_dataset import CommonDataset

class CityScapesDataset(CommonDataset):
    def _load_anno(self, seed=None):
        assert os.path.exists(self._label_path) # ground truth
        assert os.path.exists(self._data_root) # val dataset
        self.inputs = []
        self.labels = []

        self.labels = sorted(
            glob.glob(
                os.path.join(self._label_path, '*', '*_gtFine_labelTrainIds.png')))
        self.inputs = sorted(
            glob.glob(
                os.path.join(self._data_root, '*', '*_leftImg8bit.png')))