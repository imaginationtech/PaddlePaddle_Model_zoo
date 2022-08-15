from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

from utils.misc import AverageMeter
from utils import logger
from utils.logger import init_logger
from utils.config import print_config
from metric import build_metrics
from data import build_dataloader
from data import build_preprocess
from data.preprocess import transform
from data import build_postprocess
from data.utils.get_image_list import get_image_list
from .backend import build_inference

class Egret(object):
    def __init__(self, config):
        self.config = config

        self.mode = self.config['Global']['mode']
        assert self.mode in ["evaluation", "inference"], \
                "engine mode should be 'evaluation' or 'inference'"
        self.category = self.config['Global']['category']
        assert self.category in ['classification','segmentation'], \
                "nn category should be 'classification' or 'segmentation'"

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.config["Model"]["name"],
                                f"{self.mode}.log")
        init_logger(name='root', log_file=log_file)
        print_config(config)

        if self.mode == "evaluation":
            # build dataloader
            self.eval_dataloader = build_dataloader(self.config["DataLoader"])
            # build metric
            self.init_metrics()
        elif self.mode == 'inference':
            self.preprocess_ops = build_preprocess(self.config["Infer"][
                "transforms"])
            self.postprocess_func = build_postprocess(self.config["Infer"][
                "PostProcess"])

        # create runtime model
        self.inference_func = build_inference(self.config)

    def init_metrics(self):
        metric_config = self.config.get("Metric")
        if metric_config is not None:
            metric_config = metric_config.get("Eval")
            if metric_config is not None:
                self.eval_metric_func = build_metrics(metric_config)
       
    def eval(self):
        assert self.mode == "evaluation"
        assert self.inference_func is not None

        output_info = dict()
        time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        print_batch_step = self.config["Global"]["print_batch_step"]

        metric_key = None
        tic = time.time()
        total_samples = len(self.eval_dataloader.dataset)
        max_iter = len(self.eval_dataloader)
        for iter_id, batch in enumerate(self.eval_dataloader):
            if iter_id >= max_iter:
                break
            if iter_id == 5:
                for key in time_info:
                    time_info[key].reset()
            time_info["reader_cost"].update(time.time() - tic)
            batch_size = batch[0].shape[0]

            # image input
            out = self.inference_func(batch[0])

            # calc metric
            if self.eval_metric_func is not None:
                metric_dict = self.eval_metric_func(out, batch[1])
                for key in metric_dict:
                    if metric_key is None:
                        metric_key = key
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')

                    output_info[key].update(metric_dict[key],
                                            batch_size)

            time_info["batch_cost"].update(time.time() - tic)

            if iter_id % print_batch_step == 0:
                time_msg = "s, ".join([
                    "{}: {:.5f}".format(key, time_info[key].avg)
                    for key in time_info
                ])

                ips_msg = "ips: {:.5f} images/sec".format(
                    batch_size / time_info["batch_cost"].avg)
                
                #output classification result or segmentation result
                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].avg if self.category == 'classification' else output_info[key].val )
                    for key in output_info
                ])
                logger.info("[Eval][Iter: {}/{}]{}, {}, {}".format(
                    iter_id, len(self.eval_dataloader), metric_msg, time_msg,
                    ips_msg))

            tic = time.time()
        
        #output classification result or segmentation result
        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg if self.category == 'classification' 
            else output_info[key].val ) 
            for key in output_info
        ])
        logger.info("[Eval][Avg]{}".format(metric_msg))

        # do not try to save best eval.model
        if self.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return output_info[metric_key].avg

    def infer(self):
        assert self.mode == "inference" and self.category in ["classification","segmentation"]
        image_list = get_image_list(self.config["Infer"]["infer_imgs"])

        batch_size = self.config["Infer"]["batch_size"]
        batch_data = []
        image_file_list = []
        for idx, image_file in enumerate(image_list):
            with open(image_file, 'rb') as f:
                x = f.read()
            if self.preprocess_ops:
                x = transform(x, self.preprocess_ops)
            if self.category == 'segmentation':
                x = x.transpose(2, 0, 1)
            batch_data.append(x)
            image_file_list.append(image_file)
            if len(batch_data) >= batch_size or idx == len(image_list) - 1:
                batch_tensor = np.array(batch_data)
                out = self.inference_func(batch_tensor)
                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, dict) and "logits" in out:
                    out = out["logits"]
                if isinstance(out, dict) and "output" in out:
                    out = out["output"]
                if self.postprocess_func:
                    result = self.postprocess_func(out, image_file_list)
                    print(result) if self.category == 'classification' else print("Segmentaion Inference Image is saved in {} ".format(result))
                batch_data.clear()
                image_file_list.clear()

    def run(self):
        if self.mode == 'evaluation':
            self.eval()
        elif self.mode == 'inference':
            self.infer()

