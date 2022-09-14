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
from utils.box_util import bbox_ious


@op_register
class HardNMS(OpBase):
    def __init__(self, iou_threshold=0.6, threshold=0.5, max_keep=100):
        self.iou_threshold = iou_threshold
        self.threshold = threshold
        self.max_keep = max_keep

    def __call__(self, bboxes, scores, **kwargs):
        keepbboxes, keepscores, keepprds = [], [], []
        for bbox, score in zip(bboxes, scores):
            num_box, num_cls = score.shape
            keepbbox = np.array([]).reshape((0, 4))
            keepscore = np.array([])
            keeppred = np.array([])
            for cls in range(num_cls):
                cls_score = score[:, cls]
                ind = np.where(cls_score >= self.threshold)
                cls_score = cls_score[ind]
                cls_bbox = bbox[ind]
                keep = self.nms(cls_bbox, cls_score)
                keepbbox = np.vstack([keepbbox, cls_bbox[keep]])
                keepscore = np.concatenate([keepscore, cls_score[keep]])
                keeppred = np.concatenate([keeppred, np.full(shape=(len(keep),), fill_value=cls)])
            # sorted for keep
            indices = np.argsort(keepscore)[::-1]
            ind = indices[:self.max_keep]
            keepbbox = keepbbox[ind]
            keepscore = keepscore[ind]
            keeppred = keeppred[ind]

            keepbboxes.append(keepbbox)
            keepscores.append(keepscore)
            keepprds.append(keeppred)

        result = {"bboxes": keepbboxes, "scores": keepscores, "preds": keepprds}

        return result



    def nms(self, bboxes, scores):

        order = np.argsort(scores, axis=-1)[::-1]
        bboxes = bboxes[order]
        keep = []
        while order.size > 0:
            keep.append(order[0])
            bbox = bboxes[order[0:1]]
            bbox1 = bboxes[order[1:]]
            iou = bbox_ious(bbox, bbox1)
            ind = np.where(iou <= self.iou_threshold)[0]
            order = order[ind + 1]

        return keep
    