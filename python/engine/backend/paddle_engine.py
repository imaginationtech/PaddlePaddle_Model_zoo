import os
import paddle
from utils import logger

class PaddleInference(object):
    def __init__(self, paddle_config):
        device = paddle_config.get('device', 'gpu')
        paddle.set_device(device)
        logger.info('inference with paddle {} and device {}'.format(
            paddle.__version__, device))
        
        path = paddle_config.get('path', None)
        base_name = paddle_config.get('base_name', None)
        if path is not None and base_name is not None:
            model_path = os.path.join(path, base_name)
            self.inference_model = paddle.jit.load(model_path)
        else:
            self.inference_model = None

    def __call__(self, x):
        x = paddle.to_tensor(x)
        out = self.inference_model(x)
        return out.numpy()
