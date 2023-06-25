from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


coco_gld = COCO('eval/results/annotations_test.json')
coco_rst = coco_gld.loadRes('eval/results/hog_svm.json')
cocoEval = COCOeval(coco_gld, coco_rst, iouType='bbox')

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()