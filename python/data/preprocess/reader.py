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
import csv
from data.base import OpBase, op_register
from . import operators


@op_register
class LoadImage(OpBase):
    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.decode = operators.DecodeImage(to_rgb, to_np, channel_first)

    def __call__(self, images, **kwargs):
        with open(images, 'rb') as f:
            x = f.read()
        img = self.decode(x)
        result = {"images": img}

        kwargs.update(result)
        return kwargs


@op_register
class LoadKs(OpBase):
    def __init__(self, is_inv=False):
        self.is_inv = is_inv

    def __call__(self, Ks, **kwargs):
        with open(Ks, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    K = row[1:]
                    K = [float(i) for i in K]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)
                    K = K[:3, :3]
                    break
        if self.is_inv:
            K = np.linalg.inv(K)
        result = {"Ks": K}
        kwargs.update(result)

        return kwargs
