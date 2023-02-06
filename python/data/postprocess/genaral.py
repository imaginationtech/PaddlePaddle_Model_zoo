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
class Reshape(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.reshape(kwargs[k], v)
        return kwargs


@op_register("Transpose")
class Transpose(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = np.transpose(kwargs[k], v)
        return kwargs


@op_register
class ReOutputsKey(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = kwargs[v]
            kwargs.pop(v)
        return kwargs


@op_register
class ConfFiler(OpBase):
    """
    K:
        index: x
        conf: y
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            data = []
            for p in kwargs[k]:
                if p[v['index']] > v['conf']:
                    p = list(p)
                    data.append(p)
            kwargs[k] = np.array(data)

        return kwargs


@op_register
class ToKittRecord(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k in self.kwargs:
            data = []
            for p in kwargs[k]:
                p = list(p)
                p.insert(1, 0.0)
                p.insert(2, 0)
                data.append(p)
            kwargs[k] = np.array(data)

        return kwargs


@op_register
class VoxelNetDecoder(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, **kwargs):
        anchors_mask = np.load(self.kwargs['anchors_mask'])
        anchors = np.load(self.kwargs['anchors'])
        box_preds = kwargs['box_preds'].reshape([1, 107136, 7])
        cls_preds = kwargs['cls_preds'].reshape([1, 107136, 1])
        dir_preds = kwargs['dir_preds'].reshape([1, 107136, 2])
        boxes,scores,labels = self.single_post_process(box_preds, cls_preds, dir_preds, anchors_mask,anchors)
        boxes_2d = np.concatenate((boxes[:,:2],boxes[:,3:5],boxes[:,6:]),axis=-1)
        kwargs['bboxes'] = boxes_2d
        return kwargs
        
    def single_post_process(self, box_preds, cls_preds, dir_preds, anchors_mask,anchors):

        box_preds = self.second_box_decode_numpy(box_preds, anchors)
        box_preds = box_preds.squeeze(0)
        cls_preds = cls_preds.squeeze(0)
        dir_preds = dir_preds.squeeze(0).reshape((-1,2))

        # _single_post_process
        box_preds = box_preds[anchors_mask]
        cls_preds = cls_preds[anchors_mask]
        cls_confs=1/(1+(np.exp((-cls_preds))))

        cls_scores = cls_confs.max(-1)
        cls_labels = cls_confs.argmax(-1)

        kept = cls_scores >= 0.05
        dir_preds = dir_preds[anchors_mask]
        dir_labels = dir_preds.argmax(axis=-1)

        box_preds = box_preds[kept]
        cls_scores = cls_scores[kept]
        cls_labels = cls_labels[kept]
        dir_labels = dir_labels[kept]

        #_box_not_empty
        box_preds[:, 2] = box_preds[:, 2] + box_preds[:, 5] * 0.5
        for i in range(len(box_preds)):
            if box_preds[i,6] > 0:
                box_preds[i,6] += np.pi
        #rotate_nms_pcdet 
        box_preds = box_preds[:, [0, 1, 2, 4, 3, 5, -1]]
        box_preds[:, -1] = -box_preds[:, -1] - np.pi / 2
        order = cls_scores.argsort(0)[::-1]
        order = order[:1000]
        box_preds = box_preds[order]
        box_preds = box_preds.reshape([-1, 7])
        return box_preds, cls_scores, cls_labels
    
    
    def second_box_decode_numpy(self, encodings, anchors):
        """
        Decode 3D bboxes for VoxelNet/PointPillars.
        Args:
            encodings ([N, 7] Tensor): encoded boxes: x, y, z, w, l, h, r
            anchors ([N, 7] Tensor): anchors
        """
        xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = np.split(encodings, 7, axis=-1)

        diagonal = np.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)