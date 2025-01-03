import json
from typing import List, Optional

import numpy as np
from PIL import Image

from .annotation.attribute_llava import get_attributes
from .annotation.depth_estimation import depth_estimation
from .annotation.object_detection import object_detection
from .annotation.segmentation import sam_mask_generation, sam_mask_prediction


class Annotator:
	def __init__(self, device, image_list: list[str], source: list[str] = []):
		self.device = device
		self.image_list = image_list.copy()
		if len(source) == 0:
			self.source = [None] * len(image_list)
		else:
			self.source = source.copy()

	def _format_return(self, annotations: list):
		res = {}
		for i, image in enumerate(self.image_list):
			width, height = Image.open(image).size
			image_id = image.split('/')[-1].split('.')[0]
			res[image_id] = {
				"data_path" : image,
				"annotation": {
					"width" : width,
					"height": height,
					**annotations[i]
				},
				"source"    : self.source[i]
			}
		return res

	def image_attribute(
			self,
			bboxes_path: str,
			k: int = 3,
			thres: float = 0.2,
			model_name: str = "BAAI/EVA-CLIP-8B",
	):

		return self._format_return(
			get_attributes(
				self.image_list,
				bboxes_path,
				k,
				thres,
				model_name,
				self.device
			)
		)

	def depth_estimation(
			self,
			model: str = "LiheYoung/depth-anything-small-hf",
			save_path: str = "./depth_pred_res/",
			**kwargs
	):
		return self._format_return(
			depth_estimation(
				self.image_list,
				device=self.device,
				model=model,
				save_path=save_path,
				**kwargs)
		)

	def image_segmentation(
			self,
			path_to_checkpoint: str = "sam2_hiera_large.pt",
			save_path: str = "./seg_gen_res/",
			obj_det_anno_path: Optional[str] = None,
			input_points: Optional[List[np.ndarray]] = None,
			input_labels: Optional[List[np.ndarray]] = None,
			multimask_output: Optional[bool] = False,
			sam2_version: str = "072824",
			sam2_config_path: str = "sam2_hiera_l.yaml",
			save_png: bool = False,
			**kwargs
	):
		if obj_det_anno_path is not None:
			obj_det_anno = json.load(open(obj_det_anno_path, 'r'))
			bboxes = [anno['annotation']['bboxes'] for anno in obj_det_anno.values()]
			labels = [anno['annotation']['labels'] for anno in obj_det_anno.values()]
			return self._format_return(
				sam_mask_prediction(
					self.image_list,
					device=self.device,
					path_to_checkpoint=path_to_checkpoint,
					save_path=save_path,
					input_points=input_points,
					input_labels=input_labels,
					bboxes=bboxes,
					labels=labels,
					multimask_output=multimask_output,
					sam2_version=sam2_version,
					sam2_config_path=sam2_config_path,
					save_png=save_png,
					**kwargs)
			)
		else:
			return self._format_return(
				sam_mask_generation(
					self.image_list,
					device=self.device,
					path_to_checkpoint=path_to_checkpoint,
					save_path=save_path,
					sam2_version=sam2_version,
					sam2_config_path=sam2_config_path,
					save_png=save_png,
				)
			)

	def object_detection(
			self,
			model_path: str,
			model_config_path: str,
			**kwargs
	):
		return self._format_return(
			object_detection(
				self.image_list,
				model_path,
				model_config_path,
				device=self.device,
				**kwargs
			)
		)

	def relation_detection(
			self,
			clip_model_path: str,
			seg_annotation_path: str,
			seg_dir_path: str,
			osprey_model_path: str = "linxinso/osprey_relation",
			category_list_path: str = "",
			ground_pred: bool = True
	):
		from .annotation.relation_osprey import Osprey

		model = Osprey(osprey_model_path, clip_model_path, self.device)
		output = model.relation_detection(self.image_list, seg_annotation_path, seg_dir_path)
		if ground_pred:
			output = model.cate_grounding(output['relations'], category_list_path)

		return self._format_return(output)

	def attribute_detection(
			self,
			obj_det_path: str,
			model_base: str = None,
			model_path: str = "jieyuz2/llava7b-attr3",
			if_load_8bit: bool = False,
			if_load_4bit: bool = False,
			inp = "<image>\n{label}",
			conv_mode = None,
			temperature = 0.2,
			max_new_tokens = 512
		):

		attributes = get_attributes(
			self.image_list,
			model_path,
			model_base,
			obj_det_path,
			if_load_8bit,
			if_load_4bit,
			self.device,
			inp,
			conv_mode,
			temperature,
			max_new_tokens
		)

		return self._format_return(attributes)