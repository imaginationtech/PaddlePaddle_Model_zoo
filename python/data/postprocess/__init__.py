import copy
import importlib

#from . import topk

from .topk import Topk, MultiLabelTopk


def build_postprocess(config):
    config = copy.deepcopy(config)
    model_name = config.pop("name")
    mod = importlib.import_module(__name__)
    postprocess_func = getattr(mod, model_name)(**config)
    return postprocess_func
