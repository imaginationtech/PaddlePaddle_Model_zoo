import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../python')))

from data import ImageNetDataset
from data import BatchSampler
from data import DataLoader

if __name__ == "__main__":
    img_dir = '/home/jasonwang/data/imagenet_aiia/valset' 
    label_path = '/home/jasonwang/data/imagenet_aiia/valset/val.txt'
    transforms = [
            {'DecodeImage':{'to_rgb': True,'channel_first': False}},
            {'ResizeImage':{'resize_short': 256}},
            {'CropImage':{'size':224}} ]
    imagenet_dataset = ImageNetDataset(img_dir, label_path, transform_ops=transforms)
    #len = len(cocodataset)
    #a = cocodataset[10]

    bs = BatchSampler(imagenet_dataset, batch_size=10)

    eval_dataloader = DataLoader(imagenet_dataset, bs)

    for iter_id, batch in enumerate(eval_dataloader):
        print("batch shape: {}".format(batch[0].shape))
        print("label:{}".format(batch[1])) 
