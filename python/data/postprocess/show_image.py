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
from ..preprocess.reader import LoadKs, LoadImage


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
        if not isinstance(image_files, list):
            image_files = [image_files]
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


@op_register
class VisualKitt(OpBase):
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if output_dir is None:
            self.output_dir = "./output_dir"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def __call__(self, kitt_records, images, Ks, **kwargs):
        if kitt_records.size == 0:
            return
        if not isinstance(images, list):
            images = [images]
            kitt_records = [kitt_records]
        if not isinstance(Ks, list):
            Ks = [Ks]

        for kitt, image, K in zip(kitt_records, images, Ks):
            img = LoadImage()(image)["images"]
            K =LoadKs()(K)["Ks"]
            bboxes_3d = np.concatenate([kitt[:, 11:14], kitt[:, 8:11], kitt[:, 14:15]], axis=-1)
            imgpts_list = self.make_imgpts_list(bboxes_3d, K)

            img = self.draw_smoke_3d(img, imgpts_list)
            image_name = os.path.basename(image)
            image_file = os.path.join(self.output_dir, image_name)
            cv2.imwrite(image_file, img)

    def make_imgpts_list(self, bboxes_3d, K):
        """to 8 points on image"""
        # external parameters do not transform
        rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tvec = np.array([[0.0], [0.0], [0.0]])

        imgpts_list = []
        for box3d in bboxes_3d:

            locs = np.array(box3d[0:3])
            rot_y = np.array(box3d[6])

            height, width, length = box3d[3:6]
            _, box2d, box3d = self.encode_label(K, rot_y,
                                           np.array([length, height, width]), locs)

            if np.all(box2d == 0):
                continue

            imgpts, _ = cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
            imgpts_list.append(imgpts)

        return imgpts_list

    def encode_label(self, K, ry, dims, locs):
        """get bbox 3d and 2d by model output

        Args:
            K (np.ndarray): camera intrisic matrix
            ry (np.ndarray): rotation y
            dims (np.ndarray): dimensions
            locs (np.ndarray): locations
        """
        l, h, w = dims[0], dims[1], dims[2]
        x, y, z = locs[0], locs[1], locs[2]

        x_corners = [0, l, l, l, l, 0, 0, 0]
        y_corners = [0, 0, h, h, 0, 0, h, h]
        z_corners = [0, 0, 0, w, w, w, w, 0]

        x_corners += -np.float32(l) / 2
        y_corners += -np.float32(h)
        z_corners += -np.float32(w) / 2

        corners_3d = np.array([x_corners, y_corners, z_corners])
        rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]])
        corners_3d = np.matmul(rot_mat, corners_3d)
        corners_3d += np.array([x, y, z]).reshape([3, 1])

        loc_center = np.array([x, y - h / 2, z])
        proj_point = np.matmul(K, loc_center)
        proj_point = proj_point[:2] / proj_point[2]

        corners_2d = np.matmul(K, corners_3d)
        corners_2d = corners_2d[:2] / corners_2d[2]
        box2d = np.array([
            min(corners_2d[0]),
            min(corners_2d[1]),
            max(corners_2d[0]),
            max(corners_2d[1])
        ])

        return proj_point, box2d, corners_3d

    def draw_3dbbox(self, img, imgpts_list):
        connect_line_id = [
            [1, 0],
            [2, 7],
            [3, 6],
            [4, 5],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
            [0, 7],
            [7, 6],
            [6, 5],
            [5, 0],
        ]

        img_draw = img.copy()

        for imgpts in imgpts_list:
            for p in imgpts:
                p_x, p_y = int(p[0][0]), int(p[0][1])
                cv2.circle(img_draw, (p_x, p_y), 1, (0, 255, 0), -1)
            for i, line_id in enumerate(connect_line_id):

                p1 = (int(imgpts[line_id[0]][0][0]), int(imgpts[line_id[0]][0][1]))
                p2 = (int(imgpts[line_id[1]][0][0]), int(imgpts[line_id[1]][0][1]))

                if i <= 3:  # body
                    color = (255, 0, 0)
                elif i <= 7:  # head
                    color = (0, 0, 255)
                else:  # tail
                    color = (255, 255, 0)

                cv2.line(img_draw, p1, p2, color, 1)

        return img_draw

