

from data import build_dataloader
from .backend import build_inference
from data.transform import Transform

class BaseArch:
    def __init__(self, config):
        self.config = config

        self.mode = self.config['Global']['mode']
        if self.mode == "evaluation":
            self.eval_dataloader = build_dataloader(config['DataLoader'])
        self.postprocess = Transform(config['Model']["head_ops"])
        # create runtime model
        self.inference_func = build_inference(self.config)


    def infer(self):

        pass

    def statistic(self, **kwargs):
        pass

    def summarise(self):
        pass

    def eval(self):
        for i, batch in enumerate(self.eval_dataloader):
            image = batch[0]
            target = batch[1]
            out = self.inference_func(image)
            # postprocess
            out = self.postprocess(**out)
            # statistic
            self.statistic(**out, targets=target)

       # self.metrics()
        self.summarise()


    def run(self):
        if self.mode == 'evaluation':
            self.eval()
        elif self.mode == 'inference':
            self.infer()

