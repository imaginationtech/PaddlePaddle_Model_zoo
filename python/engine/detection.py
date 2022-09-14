
import numpy as np
from .common import BaseArch
from metric.cocometrics import CocoMetrics


class Detection(BaseArch):
    def __init__(self, config):
        super().__init__(config)
        self.metrics = CocoMetrics(config['Metric']['image_size'], config["Metric"]['dataset'])

    def statistic(self, bboxes, scores, preds, targets):
        self.metrics.statistic(bboxes, scores, preds, targets)

    def summarise(self):
        summ = self.metrics()
        print(summ)

