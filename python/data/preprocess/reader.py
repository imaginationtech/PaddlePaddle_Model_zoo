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
import os
import csv
from data.base import OpBase, op_register
from . import operators


@op_register
class LoadImage(OpBase):
    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.decode = operators.DecodeImage(to_rgb, to_np, channel_first)

    def __call__(self, images, **kwargs):
        with open(images, 'rb') as f:
            x = f.read()
        img = self.decode(x)
        result = {"images": img}

        kwargs.update(result)
        return kwargs

@op_register
class LoadVelodyne(OpBase):
    def __init__(self, num_point_dim):
        self.num_point_dim = num_point_dim
        
    def __call__(self, velodyne, **kwargs):
        points = np.fromfile(velodyne, np.float32).reshape(-1,self.num_point_dim)
        result = {"points": points}
        kwargs.update(result)
        return kwargs

@op_register
class LoadSemanticKITTIRange(OpBase):
    def __init__(self, proj_H, proj_W, upper_radian, lower_radian, project_label=True, labels=None):
        self.project_label = project_label
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.upper_inclination = upper_radian / 180. * np.pi
        self.lower_inclination = lower_radian / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination
        self.labels = labels
        if labels is not None:
            ## need remap_lut npy from dataloader
            remap_path = "./dataset/inference/SqueezeSegV3/remap_lut.npy"
            assert(os.path.exists(remap_path))
            self.remap_lut = np.load(remap_path)

    def __call__(self, velodyne, **kwargs):
        raw_scan = np.fromfile(velodyne, dtype=np.float32).reshape((-1, 4))
        points = raw_scan[:, 0:3]
        remissions = raw_scan[:, 3]

        # get depth of all points (L-2 norm of [x, y, z])
        depth = np.linalg.norm(points, ord=2, axis=1)

        # get angles of all points
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (
            pitch + abs(self.lower_inclination)) / self.fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_copy = np.copy(
            proj_x
        )  # save a copy in original order, for each point, where it is in the range image

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y_copy = np.copy(
            proj_y
        )  # save a copy in original order, for each point, where it is in the range image

        # unproj_range_copy = np.copy(depth)   # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # projected range image - [H,W] range (-1 is no data)
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        proj_remission = np.full((self.proj_H, self.proj_W),
                                 -1,
                                 dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices
        proj_mask = proj_idx > 0  # mask containing for each pixel, if it contains a point or not

        data = np.concatenate([
            proj_range[None, ...],
            proj_xyz.transpose([2, 0, 1]), proj_remission[None, ...]
        ])
        
        kwargs.update({"data": data})
        kwargs.update({"proj_mask": proj_mask.astype(np.float32)})

        if self.labels is not None:
            # load labels
            raw_label = np.fromfile(
                self.labels, dtype=np.uint32).reshape((-1))
            # only fill in attribute if the right size
            if raw_label.shape[0] == points.shape[0]:
                sem_label = raw_label & 0xFFFF  # semantic label in lower half
                sem_label = self.remap_lut[sem_label]
                # inst_label = raw_label >> 16  # instance id in upper half
            else:
                print("Point cloud shape: {}".format(points.shape))
                print("Label shape: {}".format(raw_label.shape))
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                    format(velodyne))
            # # sanity check
            # assert ((sem_label + (inst_label << 16) == raw_label).all())

            if self.project_label:
                # project label to range view
                # semantics
                proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                          dtype=np.int32)  # [H,W]  label
                proj_sem_label[proj_mask] = sem_label[proj_idx[proj_mask]]

                # # instances
                # proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                #                            dtype=np.int32)  # [H,W]  label
                # proj_inst_label[proj_mask] = self.inst_label[proj_idx[proj_mask]]

                self.labels = proj_sem_label.astype(np.int64)[None, ...]
            else:
                self.labels = sem_label.astype(np.int64)
            # np.save('labels.npy', self.labels) ## you can open this line to save the golden transform 
        return kwargs

@op_register
class LoadKs(OpBase):
    def __init__(self, is_inv=False):
        self.is_inv = is_inv

    def __call__(self, Ks, **kwargs):
        with open(Ks, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    K = row[1:]
                    K = [float(i) for i in K]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)
                    K = K[:3, :3]
                    break
        if self.is_inv:
            K = np.linalg.inv(K)
        result = {"Ks": K}
        kwargs.update(result)

        return kwargs
