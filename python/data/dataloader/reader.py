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

import numpy as np
from data.sampler.batch_sampler import BatchSampler
from data.dataset import Dataset#, IterableDataset


__all__ = ['DataLoader']


class DataLoader(object):

    def __init__(self,
                 dataset,
                 batch_sampler=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False):
        self.dataset = dataset

        if batch_sampler is not None:
            assert batch_size == 1 and not shuffle and not drop_last, \
                "batch_size/shuffle/drop_last should not be set when " \
                "batch_sampler is given"
            self.batch_sampler = batch_sampler
            self.batch_size = None
        elif batch_size is None:
            self.batch_sampler = None
            self.batch_size = None
        else:
            assert batch_size > 0, \
                "batch_size should be None or a positive value when " \
                "batch_sampler is not given"
            self.batch_size = batch_size
            self.batch_sampler = BatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)

        self.drop_last = drop_last
        self._sampler_iter = iter(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        batch_indices = next(self._sampler_iter)
        data = []
        label = []
        for idx in batch_indices:
            data.append(self.dataset[idx][0])
            label.append(self.dataset[idx][1])
        data_batch = self.dataset.create_inputs_batch(data)#np.array(data)
        label_batch = self.dataset.create_labels_batch(label)#np.array(label)
        return data_batch, label_batch

    def __iter__(self):
        return self

    def __call__(self):
        return self.__iter__()
