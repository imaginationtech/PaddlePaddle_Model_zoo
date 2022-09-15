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
from data.base import OpBase, op_register


@op_register
class PPYoloE(OpBase):
    def __init__(self,
                 image_size,
                 strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 ):
        self.image_size = image_size
        self.strides = strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.anchor_points, self.stride_scales = self.generate_anchors()

    def generate_anchors(self):
        anchor_points = []
        stride_scales = []
        for i, stride in enumerate(self.strides):
            h = int(self.image_size[0] / stride)
            w = int(self.image_size[1] / stride)
            shift_x = np.arange(w) + self.grid_cell_offset
            shift_y = np.arange(h) + self.grid_cell_offset
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            anchor = np.stack([shift_x, shift_y], axis=-1)
            anchor = np.reshape(anchor, newshape=(-1, 2))
            anchor_points.append(anchor)
            stride_scales.append(np.full([h * w, 1], stride))
        anchors = np.concatenate(anchor_points)
        stride_scales = np.concatenate(stride_scales)
        return anchors, stride_scales

    def __call__(self, bboxes, scores, **kwargs):
        lt, rb = np.split(bboxes, 2, -1)
        x1y1 = -lt + self.anchor_points
        x2y2 = rb + self.anchor_points

        bboxes = np.concatenate([x1y1, x2y2], axis=-1)
        bboxes *= self.stride_scales
        x1, y1, x2, y2 = np.split(bboxes, 4, axis=-1)
        x1 = np.clip(x1, 0, self.image_size[1])
        x2 = np.clip(x2, 0, self.image_size[1])
        y1 = np.clip(y1, 0, self.image_size[0])
        y2 = np.clip(y2, 0, self.image_size[0])
        bboxes = np.concatenate([x1, y1, x2, y2], axis=-1)
        result = {"bboxes": bboxes, "scores": scores}
        kwargs.update(result)
        return kwargs

