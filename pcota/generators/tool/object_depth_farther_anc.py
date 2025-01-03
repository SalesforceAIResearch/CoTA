from typing import List
import os
import numpy as np

from ..template import get_qa_template
from ...base import BaseGenerator
from ...dataset import JointAnnotation
from ..utils import *


def majority_of_array(arr, mask):
	unique, counts = np.unique(arr[mask], return_counts=True)
	if len(unique) == 1:
		return unique[0]
	return unique[np.argmax(counts)]

def avg_of_array(arr, mask, reverse=True):
	depth = arr[mask]
	if reverse:
		# depth = 1 / depth 
		# depth = (depth - depth.min()) / (depth.max() - depth.min())
		depth = depth.max() - depth

	avg_depth = np.mean(depth)
	# unique, counts = np.unique(depth, return_counts=True)
	# if len(unique) == 1:
	# 	return unique[0]
	# return unique[np.argmax(counts)]
	return float(round(avg_depth, 2))

class CompareObjectDepthGenerator(BaseGenerator):

	def __init__(self, dataset, threshold=0.1, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.threshold = threshold

	def two_object(self, annotation: JointAnnotation) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(n=2)
		depth_mask = annotation.depth.mask
		seg_mask = annotation.segment.mask

		data_list = []
		for (i, j) in n_non_overlapping_bboxes_list:
			object1, object2 = annotation.labels[i], annotation.labels[j]
			if object1 == object2:
				continue

			seg_mask1, seg_mask2 = seg_mask[i], seg_mask[j]
			if np.sum(seg_mask1) == 0 or np.sum(seg_mask2) == 0:
				continue

			# dep1, dep2 = majority_of_array(depth_mask, seg_mask1), majority_of_array(depth_mask, seg_mask2)
			dep1, dep2 = avg_of_array(depth_mask, seg_mask1), avg_of_array(depth_mask, seg_mask2)
			if abs(float(dep1) - float(dep2)) < self.threshold:
				continue

			if dep2 < dep1:
				closer, farther = annotation.labels[j], annotation.labels[i]
				closer_bbox, farther_bbox = annotation.bboxes[j], annotation.bboxes[i]
				closer_depth, farther_depth = dep2, dep1
			else:
				closer, farther = annotation.labels[i], annotation.labels[j]
				closer_bbox, farther_bbox = annotation.bboxes[i], annotation.bboxes[j]
				closer_depth, farther_depth = dep1, dep2

			if self.rng.choice([True, False]):
				labels = [closer, farther]
				bboxes = [closer_bbox, farther_bbox]
				depths = [closer_depth, farther_depth]
			else:
				labels = [farther, closer]
				bboxes = [farther_bbox, closer_bbox]
				depths = [closer_depth, farther_depth]
			height, width = annotation.height, annotation.width
			bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
			scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in labels]
   
			tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, labels)
			ann_dir = os.path.dirname(self.dataset.annotation_path)
			new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
			new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-1.jpg"
			new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)

			if self.rng.choice([True, False]):
				tool_info = {
					"tools"		: [
									{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": labels}}, 
									{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bbox": bboxes[0]}},
									{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bbox": bboxes[1]}}
								],
					"outputs"	: [
									{"image": "image-1: <image>", "regions": [{"label": labels[i], "bbox": bboxes[i], "score": scores[i]} for i in range(2)]}, 
									{"depth": float(depths[0])},
									{"depth": float(depths[1])},
								],
					"new_images": [new_image_path],
				}
			else:
				tool_info = {
					"tools"		: [
									{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "object": labels[0]}},
									{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "object": labels[1]}}
								],
					"outputs"	: [
									{"depth": float(depths[0])},
									{"depth": float(depths[1])},
								]
				}
			data_list.append((closer, farther, tool_info))

		
		return data_list

	def three_object(self, annotation: JointAnnotation) -> List:
		n_non_overlapping_bboxes_list = annotation.n_non_overlapping_bboxes(n=3)
		depth_mask = annotation.depth.mask
		seg_mask = annotation.segment.mask

		data_list = []
		for (i, j, k) in n_non_overlapping_bboxes_list:
			object1, object2, object3 = annotation.labels[i], annotation.labels[j], annotation.labels[k]
			if object1 == object2 or object1 == object3 or object2 == object3:
				continue

			seg_mask1, seg_mask2, seg_mask3 = seg_mask[i], seg_mask[j], seg_mask[k]
			if np.sum(seg_mask1) == 0 or np.sum(seg_mask2) == 0 or np.sum(seg_mask3) == 0:
				continue

			dep1, dep2, dep3 = majority_of_array(depth_mask, seg_mask1), majority_of_array(depth_mask, seg_mask2), majority_of_array(depth_mask, seg_mask3)
			sorted_items = sorted([(dep1, i), (dep2, j), (dep3, k)], key=lambda x: x[0])

			dep_diff1, dep_diff2 = abs(float(sorted_items[1][0]) - float(sorted_items[0][0])), abs(float(sorted_items[2][0]) - float(sorted_items[1][0]))

			if abs(dep_diff1 - dep_diff2) < self.threshold:
				continue

			if dep_diff1 > dep_diff2:
				if self.rng.choice([True, False]):
					anchor, closer, farther = sorted_items[-1][1], sorted_items[1][1], sorted_items[0][1]
					anchor_depth, closer_depth, farther_depth = sorted_items[-1][0], sorted_items[1][0], sorted_items[0][0]
				else:
					anchor, closer, farther = sorted_items[1][1], sorted_items[-1][1], sorted_items[0][1]
					anchor_depth, closer_depth, farther_depth = sorted_items[1][0], sorted_items[-1][0], sorted_items[0][0]
			else:
				if self.rng.choice([True, False]):
					anchor, closer, farther = sorted_items[0][1], sorted_items[1][1], sorted_items[-1][1]
					anchor_depth, closer_depth, farther_depth = sorted_items[0][0], sorted_items[1][0], sorted_items[-1][0]
				else:
					anchor, closer, farther = sorted_items[1][1], sorted_items[0][1], sorted_items[-1][1]
					anchor_depth, closer_depth, farther_depth = sorted_items[1][0], sorted_items[0][0], sorted_items[-1][0]

			anchor_label, closer_label, farther_label = annotation.labels[anchor], annotation.labels[closer], annotation.labels[farther]
			anchor_bbox, closer_bbox, farther_bbox = annotation.bboxes[anchor], annotation.bboxes[closer], annotation.bboxes[farther]
			

			if self.rng.choice([True, False]):
				labels = [anchor_label, closer_label, farther_label]
				bboxes = [anchor_bbox, closer_bbox, farther_bbox]
				depths = [anchor_depth, closer_depth, farther_depth]
			else:
				labels = [anchor_label, farther_label, closer_label]
				bboxes = [anchor_bbox, farther_bbox, closer_bbox]
				depths = [anchor_depth, farther_depth, closer_depth]

			height, width = annotation.height, annotation.width
			bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
			scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in labels]
   
			tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, labels)
			ann_dir = os.path.dirname(self.dataset.annotation_path)
			new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
			new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-1.jpg"
			new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)

			if self.rng.choice([True, False]):
				tool_info = {
					"tools"		: [
									{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": labels}}, 
									{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bbox": bboxes[0]}},
									{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bbox": bboxes[1]}},
									{"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bbox": bboxes[2]}}
								],
					"outputs"	: [
									{"image": "image-1: <image>", "regions": [{"label": labels[i], "bbox": bboxes[i], "score": scores[i]} for i in range(3)]}, 
									{"depth": float(depths[0])},
									{"depth": float(depths[1])},
									{"depth": float(depths[2])},
								],
					"new_images": [new_image_path],
				}
			else:
				tool_info = {
					"tools"		: [
									{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "object": labels[0]}},
									{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "object": labels[1]}},
									{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "object": labels[2]}},
								],
					"outputs"	: [
									{"depth": float(depths[0])},
									{"depth": float(depths[1])},
									{"depth": float(depths[2])},
								]
				}
			data_list.append((anchor_label, closer_label, farther_label, tool_info))
		return data_list


class CloserObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is closer?",
	"response": "{closer}"
	"""

	qa_templates = get_qa_template("CloserObjectGenerator")
	des_templates = [
		{
			"description": "{closer} is closer than {farther}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for closer, farther, tool_info in self.two_object(annotation):
			data_dict = {
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [closer, farther]
					}
				}
			data_dict.update(tool_info)
			data_list += self.make_one_data(
				data_dict
			)
		return data_list


class FartherObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is farther?",
	"response": "{farther}"
	"""

	qa_templates = get_qa_template("FartherObjectGenerator")
	des_templates = [
		{
			"description": "{farther} is farther than {closer}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for closer, farther, tool_info in self.two_object(annotation):
			data_dict = {
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [closer, farther]
					}
				}
			data_dict.update(tool_info)
			data_list += self.make_one_data(
				data_dict
			)
		return data_list


class CloserToAnchorObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is closer to the {anchor}?",
	"response": "{closer}"
	"""

	qa_templates = get_qa_template("CloserToAnchorObjectGenerator")
	des_templates = [
		{
			"description": "{closer} is closer to {anchor} than {farther}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for anchor, closer, farther, tool_info in self.three_object(annotation):
			data_dict = {
					"anchor"    : anchor,
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [anchor, closer, farther]
					}
				}
			data_dict.update(tool_info)
			data_list += self.make_one_data(data_dict)
		return data_list


class FartherToAnchorObjectGenerator(CompareObjectDepthGenerator):
	"""
	"prompt"  : "Which of {candidates} is farther to the {anchor}?",
	"response": "{farther}"
	"""

	qa_templates = get_qa_template("FartherToAnchorObjectGenerator")
	des_templates = [
		{
			"description": "{farther} is farther to {anchor} than {closer}."
		}
	]
	metadata: bool = True

	def _generate(self, annotation: JointAnnotation) -> List:
		data_list = []
		for anchor, closer, farther, tool_info in self.three_object(annotation):
			data_dict = {
					"anchor"    : anchor,
					"closer"    : closer,
					"farther"   : farther,
					"candidates": [closer, farther],
					"answer"    : closer,
					"metadata"  : {
						"object": [anchor, closer, farther]
					}
				}
			data_dict.update(tool_info)
			data_list += self.make_one_data(data_dict)
		return data_list


ObjectDepthFartherAncGenerator = [
	FartherToAnchorObjectGenerator
]
