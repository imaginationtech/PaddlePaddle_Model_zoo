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