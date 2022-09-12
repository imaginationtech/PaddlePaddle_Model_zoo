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


__all__ = [
    "Dataset", "IterableDataset"]


class Dataset(object):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(Dataset):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))
