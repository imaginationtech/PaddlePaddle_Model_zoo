# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
from utils import logger
from .dataset import Dataset
from ..preprocess import transform
from ..utils import create_operators


class CommonDataset(Dataset):
    def __init__(
            self,
            data_root,
            label_path,
            transform_ops=None, ):
        self._data_root = data_root
        self._label_path = label_path
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        else:
            self._transform_ops = None

        self.inputs = []
        self.labels = []
        self._load_anno()

    def _load_anno(self):
        pass

    def __getitem__(self, idx):
        try:
            with open(self.inputs[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.inputs[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.inputs)

    @property
    def class_num(self):
        return len(set(self.labels))

    @classmethod
    def create_inputs_batch(cls, data_list):
        """ Create inputs batch from input item list.

        Derived class can override this method to adapt to 
        the structure of input item
        
        Args:
            data_list: list of input items
        Returns:
            Batch created from input item list.
        """
        return np.array(data_list)

    @classmethod
    def create_labels_batch(cls, label_list):
        """ Create label batch from label item list.

        Derived class can override this method to adapt to 
        the structure of label item
        
        Args:
            label_list: list of label items
        Returns:
            Batch created from label item list.
        """
        return np.array(label_list)
