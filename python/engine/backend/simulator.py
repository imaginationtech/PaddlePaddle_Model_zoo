import numpy as np

class SimClasRuntime(object):
    def __init__(self, sim_config):
        self._class_num = sim_config['class_num']
        #self._batch_size = batch_size

    def __call__(self, x):

        batch_size = x.shape[0]

        rng = np.random.default_rng(12345)

        # generate labels for 
        #labels = rng.integers(low=0, high=self._class_num, size=self._batch_size, dtype=np.int64, endpoint=False)

        # generate inference output
        feature_list = []
        for i in range(batch_size):
           feature = rng.random(size=self._class_num, dtype=np.float32)
           feature = feature/feature.sum()
           feature_list.append(feature)
        predicts = np.array(feature_list)

        return predicts


