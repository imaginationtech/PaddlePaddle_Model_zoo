#import inspect
import copy
import numpy as np
import importlib

from utils import logger

from .dataloader import BatchSampler, DataLoader
from .dataset.imagenet_dataset import ImageNetDataset
from .dataset.cityscapes_dataset import CityScapesDataset
from .utils import create_operators
from .postprocess import build_postprocess
#from .preprocess import transform as preproc_transform


def build_preprocess(params):
    return create_operators(params)


def build_dataloader(loader_config, mode='Eval'):
    assert mode in ['Eval'], "Dataset mode should be Eval"

    mod = importlib.import_module(__name__)

    config_dataset = loader_config['Eval']['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    dataset = getattr(mod, dataset_name)(**config_dataset)

    config_sampler = loader_config['Eval']['sampler']
    assert "name" in config_sampler, \
            "sampler name should be set in config"
    config_sampler = copy.deepcopy(config_sampler)
    sampler_name = config_sampler.pop("name")
    sampler = getattr(mod, sampler_name)(dataset, **config_sampler)

    data_loader = DataLoader(dataset=dataset, batch_sampler=sampler)
    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader
