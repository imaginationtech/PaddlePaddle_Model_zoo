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
class Sigmoid(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        for k, v in self.kwargs.items():
            kwargs[k] = self.sigmoid(kwargs[v])
        return kwargs

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


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
    

@op_register
class CenterPointDecoder(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, **kwargs):
        preds_dict = {}
        for key in {"hm","height","dim","rot","reg"}:
           preds_dict[key] = kwargs[key].transpose((0, 2, 3, 1))
        test_cfg = self.kwargs['test_cfg']
        kwargs['bboxes'] = self.post_processing(preds_dict, test_cfg)
        return kwargs
    
    def post_processing(self, preds_dict, test_cfg):
        batch_hm = 1/(1+(np.exp((-preds_dict['hm']))))
        batch_dim = np.exp(preds_dict['dim'])
        batch_rots = preds_dict['rot'][..., 0:1]
        batch_rotc = preds_dict['rot'][..., 1:2]
        batch_reg = preds_dict['reg']
        batch_hei = preds_dict['height']

        batch_rot = np.arctan2(batch_rots, batch_rotc)
        batch, H, W, num_cls = batch_hm.shape

        batch_reg = batch_reg.reshape([batch, H * W, 2])
        batch_hei = batch_hei.reshape([batch, H * W, 1])

        batch_rot = batch_rot.reshape([batch, H * W, 1])
        batch_dim = batch_dim.reshape([batch, H * W, 3])
        batch_hm = batch_hm.reshape([batch, H * W, num_cls])

        xs, ys = np.meshgrid(np.arange(0, W), np.arange(0, H))

        ys = ys.reshape([1, H, W]).astype(
            batch_hm.dtype)
        xs = xs.reshape([1, H, W]).astype(
            batch_hm.dtype)

        xs = xs.reshape([batch, -1, 1]) + batch_reg[:, :, 0:1]
        ys = ys.reshape([batch, -1, 1]) + batch_reg[:, :, 1:2]
        xs = xs * test_cfg['down_ratio'] * test_cfg['voxel_size'][
            0] + test_cfg['point_cloud_range'][0]
        ys = ys * test_cfg['down_ratio'] * test_cfg['voxel_size'][
            1] + test_cfg['point_cloud_range'][1]
        batch_box_preds = np.concatenate(
                    [xs, ys, batch_hei, batch_dim, batch_rot], axis=2)

        hm_preds = batch_hm[0]
        box_preds = batch_box_preds[0]
        post_center_range = test_cfg['post_center_limit_range']
        scores = np.max(hm_preds, axis=-1)
        score_mask = scores > test_cfg['score_threshold']
        distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
            & (box_preds[..., :3] <= post_center_range[3:]).all(1)
        mask = distance_mask & score_mask
        box_preds = box_preds[mask]
        scores = scores[mask]
        order = scores.argsort(0)[::-1]
        box_preds = box_preds[order]
        return box_preds

@op_register
class Restoration(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kwargs['upper_inclination'] = self.kwargs['upper_radian'] / 180. * np.pi
        self.kwargs['lower_inclination'] = self.kwargs['lower_radian'] / 180. * np.pi
        self.kwargs['fov'] = self.kwargs['upper_inclination'] - self.kwargs['lower_inclination']

    def __call__(self, **kwargs):
        raw_scan = np.fromfile(kwargs['velodyne'], dtype=np.float32).reshape((-1, 4))
        points = raw_scan[:, 0:3]

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
            pitch + abs(self.kwargs['lower_inclination'])) / self.kwargs['fov']  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.kwargs['proj_W'] # in [0.0, W]
        proj_y *= self.kwargs['proj_H']  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.kwargs['proj_W'] - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.kwargs['proj_H'] - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        kwargs['pred_point_label'] = kwargs["pred_img_label"][0][proj_y, proj_x]
        return kwargs
    
@op_register
class VisualizePoints3D(OpBase):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, **kwargs):
        points = np.fromfile(kwargs['velodyne'], dtype=np.float32).reshape([-1, 4])
        pred_point_label = kwargs['pred_point_label']
        output_dir = self.kwargs['output_dir']
        self.viz_mayavi(points, pred_point_label, self.kwargs['bgcolor'], self.kwargs['size'], output_dir)
        return kwargs

    def viz_mayavi(self, points, label, bgcolor, size, output_dir=None):
        from mayavi import mlab
        x = points[:, 0]  # x position of point
        y = points[:, 1]  # y position of point
        z = points[:, 2]  # z position of point
        fig = mlab.figure(bgcolor= tuple(bgcolor), size=tuple(size))
        mlab.points3d(x, y, z,
                          label,          # Values used for Color
                          mode="point",
                          colormap='spectral', # 'bone', 'copper', 'gnuplot'
                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                          figure=fig,
                          )
        if output_dir is not None:
            import os
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            mlab.savefig(output_dir+"/result.png")
        else: 
            mlab.show()
        return


@op_register
class BoxLidar2Cam(OpBase):
    def __init__(self):
        pass

    def __call__(self, bboxes, lidar2cam, **kwargs):
        if not isinstance(lidar2cam, list):
            lidar2cam = [lidar2cam]
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            bbox1 = bbox.copy()
            l, w, h, r = bbox1[:, 3:4], bbox1[:, 4:5], bbox1[:, 5:6], bbox1[:, 6:7]
            r = -r - np.pi / 2
            bbox[:, 3:4] = l
            bbox[:, 4:5] = h
            bbox[:, 5:6] = w
            bbox[:, 6:7] = r

            l2c = lidar2cam[i]
            l2ct = l2c.T
            # l2ct_inv = np.linalg.inv(l2ct)
            points = bbox1[:, :3]
            pads = np.ones((bbox.shape[0], 1))
            points = np.concatenate([points, pads], axis=1)
            points[:, 2] -= h.reshape(-1) / 2
            points = points @ l2ct
            bbox[:, :3] = points[:, :3]

        result = {"bboxes": bboxes}
        kwargs.update(result)
        return kwargs


@op_register
class BBox3dToKittRecord(OpBase):
    # kitti record fields
    # type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, dh, dw, dl, lx, ly, lz, ry
    def __init__(self, **kwargs):
        pass
    def __call__(self, **kwargs):
        bboxes = kwargs['bboxes']
        kitt_records = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            t = np.zeros(shape=[bbox.shape[0], 15], dtype=np.float32)
            l, h, w, r = bbox[:, 3:4], bbox[:, 4:5], bbox[:, 5:6], bbox[:, 6:7]
            t[:, 8:9] = h
            t[:, 9:10] = w
            t[:, 10:11] = l
            t[:, 14:15] = r
            t[:, 11:14] = bbox[:, 0:3]
            kitt_records.append(t)
        result = {"kitt_records": kitt_records}
        kwargs.update(result)
        return kwargs

