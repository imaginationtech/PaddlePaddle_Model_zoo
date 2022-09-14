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
import cv2
import numpy as np
from .dataset import Dataset
from ..preprocess import transform
from ..utils import create_operators
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self,
                 image_dir=None,
                 annotation_path=None,
                 transform_ops=None,
                 num_image=None):
        super().__init__()

        self.image_dir = image_dir
        self.annotation_path = annotation_path

        if transform_ops:
            self.transform_ops = create_operators(transform_ops)
        if not os.path.exists(annotation_path):
            raise ValueError("%s is not exist" % annotation_path)
        self.coco = COCO(annotation_path)
        self.imgids = self.coco.getImgIds()
        self.catid2clsid = dict({catid: i for i, catid in enumerate(self.coco.getCatIds())})
        self.cname2cid = dict({
            self.coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if isinstance(num_image, int) and num_image <= 0 or num_image > len(self.imgids):
            raise ValueError("num image must >= or <=  %s", len(self.imgids))
        self.num_image = num_image

    def __len__(self):
        if self.num_image is None:
            return len(self.imgids)
        return self.num_image

    def __getitem__(self, item):
        img_id = self.imgids[item]
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_anno = self.coco.loadImgs([img_id])[0]
        img_fname = img_anno['file_name']
        path = os.path.join(self.image_dir, img_fname)
        w = img_anno['width']
        h = img_anno['height']
        shape = [h, w]
        bboxes = []
        labels = []
        for ann in anns:
            cat_id = ann['category_id']
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w
            y2 = y1 + h
            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)
            labels.append(self.catid2clsid[cat_id])

        gt_bbox = np.array(bboxes, dtype=np.float32)
        gt_class = np.array(labels, dtype=np.int32)
        # t = {"image": path, "gt_bbox": gt_bbox, "gt_class": gt_class}
        if self.transform_ops:
            with open(path, "rb") as f:
                img = f.read()
            image = transform(img, self.transform_ops)
        else:
            image = cv2.imread(path)

        shape = np.array(shape)
        img_id = np.array(img_id)
        return image, (gt_bbox, gt_class, shape, img_id)

    @classmethod
    def create_inputs_batch(cls, data_list):
        return np.array(data_list)

    @classmethod
    def create_labels_batch(cls, label_list):
        gt_bboxes = []
        gt_classes = []
        shapes = []
        img_ids = []
        for gt_bbox, gt_class, shape, img_id in label_list:
            gt_bboxes.append(gt_bbox)
            gt_classes.append(gt_class)
            shapes.append(shape)
            img_ids.append(img_id)

        return gt_bboxes, gt_classes, shapes, img_ids
