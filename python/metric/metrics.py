import numpy as np
from sklearn.metrics import top_k_accuracy_score


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
