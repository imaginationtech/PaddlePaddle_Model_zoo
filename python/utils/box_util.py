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

def bbox_ious(bbox1, bbox2):
    x1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    y1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    x2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    y2 = np.minimum(bbox1[:, 3], bbox2[:, 3])
    h = np.maximum(y2 - y1, 0)
    w = np.maximum(x2 - x1, 0)
    overlap = h * w

    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    area = area1 + area2 - overlap
    iou = overlap / area
    return iou

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def adjust_bbox(bboxes, shapes, input_size=(640, 640)):
    hi, wi = input_size
    bboxes_ = []
    for bbox, shape in zip(bboxes, shapes):
        h, w = shape
        x1, y1, x2, y2 = np.split(bbox, 4, axis=-1)
        x1 = x1 / wi * w
        x2 = x2 / wi * w
        y1 = y1 / hi * h
        y2 = y2 / hi * h
        bbox = np.concatenate([x1, y1, x2, y2], axis=-1)
        bboxes_.append(bbox)
    return bboxes_

