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

import os.path
import cv2
import yaml
import random
import numpy as np
from data.base import OpBase, op_register


@op_register
class ShowImage(OpBase):
    def __init__(self, image_size, cls_path, output_dir=None):
        self.image_size = image_size
        self.cls_path = cls_path
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = "./output_dir"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        with open(cls_path) as f:
            names = yaml.load(f, yaml.Loader)["names"]
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        self.names = names

    def __call__(self, bboxes, scores, preds, image_files, **kwargs):
        for bbox, score, pred, img_path in zip(bboxes, scores, preds, image_files):
            img = cv2.imread(img_path)
            h, w, c = img.shape
            hi, wi = self.image_size
            for i in range(bbox.shape[0]):
                label = int(pred[i])
                sco = score[i]
                box = bbox[i].astype(np.int32)
                c1, c2 = (int(box[0] / wi * w), int(box[1] / hi * h)), (int(box[2] / wi * w), int(box[3] / hi * h))
                color = self.colors[label]
                name = self.names[label]

                tl = 1
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)  # font thickness
                name = name + ": %.2f" % sco
                t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                            lineType=cv2.LINE_AA)
            image_name = os.path.basename(img_path)
            image_file = os.path.join(self.output_dir, image_name)
            cv2.imwrite(image_file, img)
        result = {'bboxes': bboxes, 'scores': scores, 'preds': preds, 'image_files': image_files}
        kwargs.update(result)
        return kwargs
