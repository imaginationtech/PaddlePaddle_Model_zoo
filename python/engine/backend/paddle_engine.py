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

import os
import paddle
from utils import logger

class PaddleInference(object):
    def __init__(self, paddle_config):
        device = paddle_config.get('device', 'gpu')
        paddle.set_device(device)
        logger.info('inference with paddle {} and device {}'.format(
            paddle.__version__, device))
        
        path = paddle_config.get('path', None)
        base_name = paddle_config.get('base_name', None)
        if path is not None and base_name is not None:
            model_path = os.path.join(path, base_name)
            self.inference_model = paddle.jit.load(model_path)
        else:
            self.inference_model = None

    def __call__(self, x):
        x = paddle.to_tensor(x)
        out = self.inference_model(x)
        return out.numpy()
