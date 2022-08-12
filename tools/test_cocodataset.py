import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../python')))

from data import COCODataset

if __name__ == "__main__":
    img_dir = '/home/jasonwang/data/coco/val2017' 
    annotation_path = '/home/jasonwang/data/coco/annotations/instances_val2017.json'
    cocodataset = COCODataset(img_dir, annotation_path)
    len = len(cocodataset)
    a = cocodataset[10]
