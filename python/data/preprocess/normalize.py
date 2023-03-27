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
from data.base import OpBase, op_register
from . import operators


@op_register
class NormalizeImage(OpBase):
    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        self.normalize = operators.NormalizeImage(scale, mean, std, order, output_fp16, channel_num)

    def __call__(self, images, **kwargs):
        img = self.normalize(images)
        result = {"images": img}

        kwargs.update(result)
        return kwargs
    

@op_register
class NormalizeRangeImage(OpBase):
    def __init__(self,
                 mean=None,
                 std=None):
        if not (isinstance(mean,
                           (list, tuple)) and isinstance(std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        self.mean = mean
        self.std = std

    def __call__(self, samples, **kwargs):
        self.mean = np.array(self.mean).reshape(5,1,1).astype('float32')
        self.std = np.array(self.std).reshape(5,1,1).astype('float32')

        samples -= self.mean
        samples /= self.std
        result = {"samples": samples}

        kwargs.update(result)
        return kwargs

