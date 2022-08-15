import copy
from collections import OrderedDict
import importlib

from .metrics import TopkAcc, mIou


class CombinedMetrics(object):
    def __init__(self, config_list):
        super().__init__()
        self.metric_func_list = []
        assert isinstance(config_list, list), (
            'operator config should be a list')
        mod = importlib.import_module(__name__)
        for config in config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            metric_name = list(config)[0]
            metric_params = config[metric_name]
            if metric_params is not None:
                self.metric_func_list.append(
                    getattr(mod, metric_name)(**metric_params))
                    #eval(metric_name)(**metric_params))
            else:
                #self.metric_func_list.append(eval(metric_name)())
                self.metric_func_list.append(getattr(mod, metric_name)())

    def __call__(self, *args, **kwargs):
        metric_dict = OrderedDict()
        for idx, metric_func in enumerate(self.metric_func_list):
            metric_dict.update(metric_func(*args, **kwargs))
        return metric_dict


def build_metrics(config):
    metrics_list = CombinedMetrics(copy.deepcopy(config))
    return metrics_list
