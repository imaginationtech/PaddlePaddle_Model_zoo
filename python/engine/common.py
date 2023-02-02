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

import cv2
import numpy as np
from data import build_dataloader
from .backend import build_inference
from data.transform import Transform
from data.utils.get_image_list import get_image_list, get_infer_datas
from data.preprocess import transform
from data.utils import create_operators

class BaseArch:
    def __init__(self, config):
        self.config = config

        self.mode = self.config['Global']['mode']
        if self.mode == "evaluation":
            self.eval_dataloader = build_dataloader(config['DataLoader'])
        self.postprocess = Transform(config['Model'].get("head_ops", None))
        # create runtime model
        self.inference_func = build_inference(self.config)

    def infer(self):
        # image_list = get_image_list(self.config["Infer"]["infer_imgs"])
        # preprocess_ops = self.config['Infer'].get('transform_ops', None)
        # preprocess_ops = create_operators(preprocess_ops)
        # postprocess_ops = self.config['Infer'].get('postprocess_ops', None)
        # postprocess = self.postprocess + Transform(postprocess_ops)
        # for i, image_file in enumerate(image_list):
        #     if preprocess_ops:
        #         with open(image_file, 'rb') as f:
        #             x = f.read()
        #             x = transform(x, preprocess_ops)
        #     else:
        #         x = cv2.imread(image_file)
        #     x = np.array([x])
        #     out = self.inference_func(x)
        #     postprocess(**out, image_files=[image_file])

        infer_data = get_infer_datas(self.config['Infer']['infer_datas'])
        preprocess_ops = self.config['Infer'].get('transform_ops', None)
        preprocess = Transform(preprocess_ops)
        postprocess_ops = self.config['Infer'].get('postprocess_ops', None)
        postprocess = self.postprocess + Transform(postprocess_ops)
        for i, datas in enumerate(infer_data):
            x = preprocess(**datas)
            out = self.inference_func(x)
            out.update(datas)
            postprocess(**out)


    def eval(self):
        for i, batch in enumerate(self.eval_dataloader):
            image = batch[0]
            target = batch[1]
            out = self.inference_func(image)
            # postprocess
            out = self.postprocess(**out)
            # statistic
            self.statistic(**out, targets=target)

       # self.metrics()
        self.summarise()

    def run(self):
        if self.mode == 'evaluation':
            self.eval()
        elif self.mode == 'inference':
            self.infer()

    def statistic(self, **kwargs):
        pass

    def summarise(self):
        pass
