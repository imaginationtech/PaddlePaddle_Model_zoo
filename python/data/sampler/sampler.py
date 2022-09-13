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
from __future__ import division

import numpy as np


__all__ = [
    "Sampler", "SequenceSampler", "RandomSampler"
]


class Sampler(object):
    """
    An abstract class to encapsulate methods and behaviors of samplers.

    Returns:
        Sampler: an iterable object for sample indices iterating
    """

    def __init__(self, data_source=None):
        self.data_source = data_source

    def __iter__(self):
        raise NotImplementedError


class SequenceSampler(Sampler):
    """
    Iterate samples sequentially, yield :code:`0, 1, 2, ..., len(data_source) -1`
    generally,

    Returns:
        Sampler: a Sampler yield sample index sequentially
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """
    Iterate samples randomly, yield shuffled indices, if :attr:`replacement=False`,
    yield shuffled indices of the whole data souce, if :attr:`replacement=True`,
    :attr:`num_samples` can set to specify the sample number to draw.

    Args:
        data_source(Dataset): dataset to sample, this could be an
                instance of :code:`Dataset` other Python
                object which implemented :code:`__len__`.
        replacement(bool): If False, sample the whole dataset, If False,
                set :attr:`num_samples` for how many sample to draw. Default False.
        num_samples(int): set sample number to draw if :attr:`replacement`
                is True. Default None.
        generator(Generator): specify a generator to sample the data source. Default None
        
    Returns:
        Sampler: a Sampler yield sample index randomly
    """

    def __init__(self,
                 data_source,
                 replacement=False,
                 num_samples=None,
                 generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("expect boolean value for replacement, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "num_samples should not be specified while replacement is False")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer, "
                             "but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.generator:
            for i in range(self.num_samples):
                try:
                    index = next(self.generator)
                except StopIteration:
                    return
                yield index
        else:
            if self.replacement:
                for index in np.random.choice(
                        np.arange(n), self.num_samples, replace=True).tolist():
                    yield index
            else:
                for index in np.random.choice(
                        np.arange(n), n, replace=False).tolist():
                    yield index

    def __len__(self):
        return self.num_samples



