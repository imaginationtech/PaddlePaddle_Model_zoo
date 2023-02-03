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


from .detection import Detection
from .detection3d import Detection3D

MODEL_MAP = {
    'detection': Detection,
    'detection3d': Detection3D
}

def build(config):
    Model = MODEL_MAP.get(config['Global']['category'], None)
    if Model is None:
        raise ValueError('model category %s is not supported' % config['Global']['category'])
    return Model(config)
