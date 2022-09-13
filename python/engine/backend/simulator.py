# Copyright (c) 2022 Imagination Technologies Ltd. 
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

import numpy as np

class SimClasRuntime(object):
    def __init__(self, sim_config):
        self._class_num = sim_config['class_num']
        #self._batch_size = batch_size

    def __call__(self, x):

        batch_size = x.shape[0]

        rng = np.random.default_rng(12345)

        # generate labels for 
        #labels = rng.integers(low=0, high=self._class_num, size=self._batch_size, dtype=np.int64, endpoint=False)

        # generate inference output
        feature_list = []
        for i in range(batch_size):
           feature = rng.random(size=self._class_num, dtype=np.float32)
           feature = feature/feature.sum()
           feature_list.append(feature)
        predicts = np.array(feature_list)

        return predicts


