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
import cv2
from data.base import OpBase, op_register
from . import operators


@op_register
class BatchData(OpBase):
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        for k, value in kwargs.items():
            kwargs[k] = np.array([value])

        return kwargs


@op_register
class ResizeImage(OpBase):
    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="pil"):
        self.resize = operators.ResizeImage(size, resize_short, interpolation, backend)

    def __call__(self, images, **kwargs):
        img = self.resize(images)
        result = {"images": img}

        kwargs.update(result)
        return kwargs


@op_register
class ComputeDownRatio(OpBase):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, down_ratios, **kwargs):
        img = cv2.imread(down_ratios)
        h, w, _ = img.shape
        fh, fw = self.output_size
        h_ratio = h / fh
        w_ratio = w / fw
        down_ratios = np.array([h_ratio, w_ratio], dtype=np.float32)

        result = {"down_ratios": down_ratios}
        kwargs.update(result)
        return kwargs

@op_register
class HardVoxelize(OpBase):
    def __init__(self, point_cloud_range, voxel_size, max_points_in_voxel,
                 max_voxel_num):
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.max_points_in_voxel = max_points_in_voxel
        self.max_voxel_num = max_voxel_num

    def __call__(self, points, **kwargs):
        num_points, num_point_dim = points.shape[0:2]
        self.point_cloud_range = np.array(self.point_cloud_range)
        self.voxel_size = np.array(self.voxel_size)
        voxels = np.zeros((self.max_voxel_num, self.max_points_in_voxel, num_point_dim),
                        dtype=points.dtype)
        coords = np.zeros((self.max_voxel_num, 3), dtype=np.int32)
        num_points_per_voxel = np.zeros((self.max_voxel_num, ), dtype=np.int32)
        grid_size = np.round((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) /
                            self.voxel_size).astype('int32')

        grid_size_x, grid_size_y, grid_size_z = grid_size

        grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                        -1,
                                        dtype=np.int32)

        num_voxels = self._points_to_voxel(points, self.voxel_size, self.point_cloud_range,
                                    grid_size, voxels, coords,
                                    num_points_per_voxel, grid_idx_to_voxel_idx,
                                    self.max_points_in_voxel, self.max_voxel_num)

        voxels = voxels[:num_voxels]
        coords = coords[:num_voxels]
        num_points_per_voxel = num_points_per_voxel[:num_voxels]
        coords = (np.append(np.zeros((coords.shape[0],1)), coords, axis=-1)).astype('float32')
        num_points_per_voxel = num_points_per_voxel.astype('float32')
        result = {"coords": coords[:,2:],
                  "num_points_per_voxel":num_points_per_voxel,
                  "voxels":voxels}
        kwargs.update(result)
        return kwargs

    def _points_to_voxel(self, points, voxel_size, point_cloud_range, grid_size, voxels,  
                     coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                     max_points_in_voxel, max_voxel_num):
        num_voxels = 0
        num_points = points.shape[0]
        # x, y, z
        coord = np.zeros(shape=(3, ), dtype=np.int32)

        for point_idx in range(num_points):
            outside = False
            for i in range(3):
                coord[i] = np.floor(
                    (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
                if coord[i] < 0 or coord[i] >= grid_size[i]:
                    outside = True
                    break
            if outside:
                continue
            voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
            if voxel_idx == -1:
                voxel_idx = num_voxels
                if num_voxels >= max_voxel_num:
                    continue
                num_voxels += 1
                grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
                coords[voxel_idx, 0:3] = coord[::-1]
            curr_num_point = num_points_per_voxel[voxel_idx]
            if curr_num_point < max_points_in_voxel:
                voxels[voxel_idx, curr_num_point] = points[point_idx]
                num_points_per_voxel[voxel_idx] = curr_num_point + 1

        return num_voxels

@op_register
class MaskAndExpand(OpBase):
    def __init__(self, expand_dim):
        self.expand_dim = expand_dim

    def __call__(self, data, **kwargs):
        if 'proj_mask' in kwargs:
            data *= kwargs['proj_mask']
            del kwargs['proj_mask']
        data = np.expand_dims(data, self.expand_dim)
        kwargs.update({"data": data})
        return kwargs


@op_register
class CalibCamera2Img(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, calib, **kwargs):
        result = {"camera2img": calib[2], "calib": calib}
        kwargs.update(result)

        return kwargs


@op_register
class CalibLidar2Cam(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, calib, **kwargs):
        r0 = calib[4]  # R0_rect
        v2c = calib[5]
        v2c = np.vstack((v2c, np.array([0, 0, 0, 1],
                                       dtype=np.float32)))  # (4, 4)
        r0 = np.hstack((r0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        r0 = np.vstack((r0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
        v2r = r0 @ v2c
        result = {"lidar2cam": v2r, "calib": calib}
        kwargs.update(result)
        return kwargs


@op_register
class CalibLoadKs(OpBase):
    def __init__(self, is_inv=False, is_full=False):
        self.is_inv = is_inv
        self.is_full = is_full

    def __call__(self, **kwargs):
        calib = kwargs["calib"]
        K = calib[2]
        if not self.is_full:
            K = K[:3, :3]
        if self.is_inv:
            K = np.linalg.inv(K)
        result = {"Ks": K}
        kwargs.update(result)

        return kwargs


@op_register
class RemoveKeyItems(OpBase):
    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, **kwargs):
        if self.keys is not None:
            for key in self.keys:
                if key in kwargs:
                    kwargs.pop(key)
        return kwargs
