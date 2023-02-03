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


@op_register
class Reshape(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.reshape(kwargs[k], v)
        return kwargs


@op_register("Transpose")
class Transpose(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.transpose(kwargs[k], v)
        return kwargs


@op_register
class ReOutputsKey(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = kwargs[v]
            kwargs.pop(v)
        return kwargs


@op_register
class ConfFiler(OpBase):
    """
    K:
        index: x
        conf: y
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            data = []
            for p in kwargs[k]:
                if p[v['index']] > v['conf']:
                    p = list(p)
                    data.append(p)
            kwargs[k] = np.array(data)

        return kwargs


@op_register
class ToKittRecord(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k in self.kwargs:
            data = []
            for p in kwargs[k]:
                p = list(p)
                p.insert(1, 0.0)
                p.insert(2, 0)
                data.append(p)
            kwargs[k] = np.array(data)

        return kwargs
