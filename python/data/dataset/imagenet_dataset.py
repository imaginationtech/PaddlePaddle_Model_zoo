from __future__ import print_function

import numpy as np
import os

from .common_dataset import CommonDataset


class ImageNetDataset(CommonDataset):
    def _load_anno(self, seed=None):
        assert os.path.exists(self._label_path)
        assert os.path.exists(self._data_root)
        self.inputs = []
        self.labels = []

        with open(self._label_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(" ")
                self.inputs.append(os.path.join(self._data_root, l[0]))
                self.labels.append(np.int64(l[1]))
                assert os.path.exists(self.inputs[-1])
