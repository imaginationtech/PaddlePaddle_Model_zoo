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

from data import build_dataloader
from .backend import build_inference
from data.transform import Transform

class BaseArch:
    def __init__(self, config):
        self.config = config

        self.mode = self.config['Global']['mode']
        if self.mode == "evaluation":
            self.eval_dataloader = build_dataloader(config['DataLoader'])
        self.postprocess = Transform(config['Model']["head_ops"])
        # create runtime model
        self.inference_func = build_inference(self.config)


    def infer(self):

        pass

    def statistic(self, **kwargs):
        pass

    def summarise(self):
        pass

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

