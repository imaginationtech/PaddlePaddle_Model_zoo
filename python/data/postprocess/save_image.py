import numpy as np
from PIL import Image as PILImage
import os

class SaveImages(object):
    def __init__(self, save_path='output/'):
        assert isinstance(save_path, str)
        assert os.path.exists(save_path)
        self.save_path = save_path

    def get_pseudo_color_map(self, pred, color_map=None):
    
        pred = np.squeeze(pred)
        pred_mask = PILImage.fromarray(np.array(pred,dtype=np.uint8), mode='P')
        if color_map is None:
            color_map = self.get_color_map_list(256)
        pred_mask.putpalette(color_map)
        return pred_mask

    def get_color_map_list(self, num_classes, custom_color=None):
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[:len(custom_color)] = custom_color
        return color_map

    def __call__(self, results, imgs_path):

        for i in range(results.shape[0]):
            result = self.get_pseudo_color_map(results)
            basename = os.path.join(self.save_path, imgs_path[-1].split('/')[-1])
            result.save(basename)
            return basename