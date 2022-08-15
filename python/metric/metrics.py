import numpy as np
import os
from PIL import Image
from sklearn.metrics import top_k_accuracy_score

class mIou(object):
    def __init__(self, num_classes=19):
        super().__init__()
        assert isinstance(num_classes, (int))
        self.num_classes = num_classes
        # The intersection area of prediction and the ground on all class.
        self.intersect_area_all = np.zeros([1], dtype='int64') 
        # The prediction area on all class.
        self.pred_area_all = np.zeros([1], dtype='int64') 
        # The ground truth area on all class
        self.label_area_all = np.zeros([1], dtype='int64') 

    def calculate_area(self, pred, label, num_classes=19, ignore_index=255):
        """
        Calculate intersect, prediction and label area
        """
        if not pred.shape == label.shape:
            raise ValueError('Shape of `pred` and `label should be equal, '
                            'but there are {} and {}.'.format(pred.shape,
                                                            label.shape))
        pred_area = []
        label_area = []
        intersect_area = []

        mask = label != ignore_index
        for i in range(num_classes):
            pred_i = np.logical_and(pred == i, mask)
            label_i = label == i
            intersect_i = np.logical_and(pred_i, label_i)
            pred_area.append(np.sum(np.cast["int64"](pred_i)))
            label_area.append(np.sum(np.cast["int64"](label_i)))
            intersect_area.append(np.sum(np.cast["int64"](intersect_i)))

        self.intersect_area_all =  self.intersect_area_all + np.stack(intersect_area)
        self.pred_area_all =  self.pred_area_all + np.stack(pred_area)
        self.label_area_all = self.label_area_all + np.stack(label_area)
    
    def mean_iou(self, intersect_area, pred_area, label_area):
        """
        Calculate iou.
        Returns:
            float: mean iou of all classes.
        """

        union = pred_area + label_area - intersect_area
        class_iou = []
        for i in range(len(intersect_area)):
            if union[i] == 0:
                iou = 0
            else:
                iou = intersect_area[i] / union[i]
            class_iou.append(iou)
        miou = np.mean(class_iou)
        return miou

    def __call__(self, out, label):
        assert isinstance(label[0], (str))
        assert os.path.exists(label[0])
        np_label = np.array([np.asarray(Image.open(label[0]))]) # Ground Truth is .png file path
        metric_dict = dict()
        self.calculate_area(out, np_label, num_classes = self.num_classes)
        metric_dict['miou'] = self.mean_iou(self.intersect_area_all, self.pred_area_all, self.label_area_all)
        return metric_dict


class TopkAcc(object):
    def __init__(self, topk=(1, 5)):
        super().__init__()
        assert isinstance(topk, (int, list, tuple))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def _top_k_accuracy_score(self, y_score, y_true, k=1, normalize=True):
        assert y_score.ndim==2, "y_score should be 2-dims"

        n_classes = y_score.shape[1]
        assert y_true.max() <= n_classes, "invalid y_true: {} v.s {}".format(y_true.max(),n_classes)
        assert k <= n_classes, "invalid k"
        assert y_score.shape[0] == y_true.shape[0], "invalid batch"

        sorted_pred = np.argsort(y_score, axis=1, kind='mergesort')[:, ::-1]
        hits = (y_true == sorted_pred[:, :k].T).any(axis=0)

        if normalize:
            return np.mean(hits)
        else:
            return np.sum(hits)

    def __call__(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]

        metric_dict = dict()
        for k in self.topk:
            metric_dict["top{}".format(k)] = self._top_k_accuracy_score(
                x, label, k=k)
        return metric_dict
