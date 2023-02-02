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
import cv2
from data.base import OpBase, op_register
from . import operators


@op_register
class BatchData(OpBase):
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        for k, value in kwargs.items():
            kwargs[k] = np.array([value])

        return kwargs


@op_register
class ResizeImage(OpBase):
    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="pil"):
        self.resize = operators.ResizeImage(size, resize_short, interpolation, backend)

    def __call__(self, images, **kwargs):
        img = self.resize(images)
        result = {"images": img}

        kwargs.update(result)
        return kwargs


@op_register
class ComputeDownRatio(OpBase):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, down_ratios, **kwargs):
        img = cv2.imread(down_ratios)
        h, w, _ = img.shape
        fh, fw = self.output_size
        h_ratio = h / fh
        w_ratio = w / fw
        down_ratios = np.array([h_ratio, w_ratio], dtype=np.float32)

        result = {"down_ratios": down_ratios}
        kwargs.update(result)
        return kwargs
