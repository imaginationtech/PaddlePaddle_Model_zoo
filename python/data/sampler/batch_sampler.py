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
import math

from data.dataset import Dataset, IterableDataset
from .sampler import Sampler, SequenceSampler, RandomSampler

__all__ = ["BatchSampler"]


class BatchSampler(Sampler):

    def __init__(self,
                 dataset=None,
                 sampler=None,
                 shuffle=False,
                 batch_size=1,
                 drop_last=False):
        if dataset is None:
            assert sampler is not None, \
                "either dataset or sampler should be set"
            assert isinstance(sampler, Sampler), \
                "sampler should be a Sampler, but got {}".format(type(sampler))
            assert not shuffle, "shuffle should be False when sampler is set"
            self.sampler = sampler
        else:
            assert not isinstance(dataset, IterableDataset), \
                "dataset should not be a IterableDataset"
            assert sampler is None, \
                "should not set both dataset and sampler"
            assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value, but got {}".format(type(shuffle))
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequenceSampler(dataset)

        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size should be a positive integer, but got {}".format(batch_size)
        self.batch_size = batch_size
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean value, but got {}".format(type(drop_last))
        self.drop_last = drop_last

    def __iter__(self):
        batch_indices = []
        for idx in self.sampler:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = len(self.sampler)
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
