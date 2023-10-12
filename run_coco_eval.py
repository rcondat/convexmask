"""
Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.
"""

from data import *

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--config', type=str)
parser.add_argument('--det_folder', default='results/', type=str)
parser.add_argument('--gt_ann_file',   default='data/coco/annotations/instances_val2017.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str)
args = parser.parse_args()



if __name__ == '__main__':
        
	if args.config is not None:
		set_cfg(args.config)
	
	bbox_det_file = args.det_folder + cfg.name + '/bbox_detections.json'
	mask_det_file = args.det_folder + cfg.name + '/mask_detections.json'

	eval_bbox = (args.eval_type in ('bbox', 'both'))
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
	
	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()



