
from .base import MetricBase
from utils.box_util import xyxy2xywh, adjust_bbox

class CocoMetrics(MetricBase):
    def __init__(self, image_size, dataset):
        super().__init__()
        self.image_size = image_size
        self.jdict = []
        self.image_ids = []
        self.dataset = dataset

    def statistic(self, bboxes, scores, preds, targets):
        gt_boxes, gt_preds, shapes, image_ids = targets
        self.image_ids.extend(image_ids)
        coco91class = self.coco80_to_coco91_class()
        bboxes = adjust_bbox(bboxes, shapes, self.image_size)
        for i, (bbox, score, pred, image_id) in enumerate(zip(bboxes, scores, preds, image_ids)):
            box = xyxy2xywh(bbox)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b, s in zip(pred.tolist(), box.tolist(), score.tolist()):
                self.jdict.append({'image_id': int(image_id),
                                   'category_id': coco91class[int(p)],
                                   'bbox': [round(x, 3) for x in b],
                                   'score': round(s, 5)})

    def coco80_to_coco91_class(self):  # converts 80-index (val2014) to 91-index (paper)
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def __call__(self, *args, **kwargs):
        import json
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        file = "detections_val2017_result.json"
        json_str = json.dumps(self.jdict)
        with open(file, 'w') as f:
            f.write(json_str)

        imgIds = self.image_ids
        cocoGt = COCO(self.dataset)  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes(str(file))  # initialize COCO pred api
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # image IDs to evaluate
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        return map, map50



