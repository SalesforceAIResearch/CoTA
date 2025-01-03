from collections import Counter, defaultdict
from typing import List

import inflect

from ..attribute_classifier import AttributeClassifier
from ..template import get_qa_template
from ..utils import *
from ...base import BaseGenerator
from ...dataset import Attributes, JointDataset


class ExistsAttributeGenerator(BaseGenerator):
	"""
	"prompt"  : "How many {name} are there?",
	"response": "{count}"
	"""

	qa_templates = get_qa_template("ExistsAttributeGenerator")
	des_templates = [
		{
			"description": "There {be} {count_name}."
		}
	]
	inflect_engine = inflect.engine()
	metadata: bool = True

	def __init__(self, dataset: JointDataset, number_mode="random", **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.number_mode = number_mode

	def _generate(self, annotation: Attributes) -> List:
		data_list = []

		attributed_labels = []
		attr_obj_to_bboxes = defaultdict(list)
		for attr, label, bbox in zip(annotation.attributes, annotation.labels, annotation.bboxes):
			if len(attr):
				attributed_obj = (tuple(sorted(attr)), label)
				attributed_labels.append(attributed_obj)
				attr_obj_to_bboxes[attributed_obj].append(bbox)

		for (attr, obj), cnt in Counter(attributed_labels).items():
			if cnt == 1:
				candidates = [1, 2, 3, 4]
			else:
				candidates = [cnt, cnt - 1, cnt + 1, cnt + 2]
			candidates = get_cnt_word(candidates, self.rng, self.inflect_engine, self.number_mode)
			cnt_word = candidates[0]
			
			be = 'is'
			attr_desc = make_and_description(attr, self.rng)
			name = f'{attr_desc} {obj}'

			if check_object_for_counting_task(obj):
				if cnt > 1:
					be = 'are'
					name = f'{attr_desc} {self.inflect_engine.plural(obj)}'
				count_name = cnt_word + ' ' + name
			else:
				if self.template_mode == 'qa':
					continue
				be = 'are'
				cnt_word = ''
				count_name = name

			bboxes = attr_obj_to_bboxes[(attr, obj)]
			assert len(bboxes) == cnt
			height, width = annotation.height, annotation.width
			bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
			scores = [round(self.rng.uniform(0.50, 1.00), 2) for _ in bboxes]

			# tagging and saving the tagged image
			tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [name] * cnt)
			ann_dir = os.path.dirname(self.dataset.annotation_path)
			new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
			new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-1.jpg"
			new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
   
			data_list += self.make_one_data(
				{
					"name"      : name,
					"count_name": count_name,
					"be"        : be,
					"count"     : cnt_word,
					"candidates": candidates,
					"answer"    : cnt_word,
					"metadata"  : {
						"object"   : [obj],
						"attribute": attr,
						"count"    : [cnt],
					},
					"tools"		: [{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": [name]}}],
					"outputs"	: [{"image": "image-1: <image>",
                   					"regions": [{"label": name, "bbox": bboxes[i], "score": scores[i]} for i in range(cnt)]}],
					"new_images": [new_image_path]
				},
			)
		return data_list


class AttributeBBoxGenerator(BaseGenerator):
	"""
	"prompt"  : "What are attributes of {name} at region {bbox}?",
	"response": "{attribute_values}"
	"""

	qa_templates = get_qa_template("AttributeBBoxGenerator")
	des_templates = [
		{
			"description": "The {name} at region {bbox} is {attribute_values}."
		}
	]
	attribute_classifier = AttributeClassifier()
	metadata: bool = True

	def _generate(self, annotation: Attributes) -> List:
		data_list = []
		height, width = annotation.height, annotation.width

		annotation = annotation.attributed_bboxes()

		for name, bbox, attributes in zip(annotation.labels, annotation.bboxes, annotation.attributes):
			candidates = self.attribute_classifier.sample_attribute(self.rng, n=self.n_choice - 1, exclude=attributes)
			answer = self.rng.choice(attributes)
			candidates = [answer] + candidates
			data_list += self.make_one_data(
				{
					"name"            : name,
					"bbox"            : str(bbox_coordinate_to_ratio(bbox, height, width)),
					"attribute_values": attributes,
					"candidates"      : candidates,
					"answer"          : answer,
					"metadata"        : {
						"object"   : [name],
						"attribute": attributes
					}
				},
			)

		return data_list


class TypedAttributeBBoxGenerator(BaseGenerator):
	"""
	"prompt"  : "What are {attribute_type} of {name} at region {bbox}?",
	"response": "{attribute_values}"
	"""

	qa_templates = get_qa_template("TypedAttributeBBoxGenerator")
	attribute_classifier = AttributeClassifier()
	metadata: bool = True

	def _generate(self, annotation: Attributes) -> List:
		data_list = []
		height, width = annotation.height, annotation.width

		annotation = annotation.attributed_bboxes()

		for name, bbox, attributes in zip(annotation.labels, annotation.bboxes, annotation.attributes):

			type_to_attributes = {}
			for attr in attributes:
				attribute_type = self.attribute_classifier.classify(attr)
				if attribute_type not in type_to_attributes:
					type_to_attributes[attribute_type] = []
				type_to_attributes[attribute_type].append(attr)

			for attribute_type, attr in type_to_attributes.items():
				if attribute_type != 'other':
					candidates = self.attribute_classifier.sample_attribute_from_category(attribute_type, self.rng, n=self.n_choice - 1, exclude=attr)
					answer = self.rng.choice(attr)
					candidates = [answer] + candidates
					data_list += self.make_one_data(
						{
							"name"            : name,
							"bbox"            : str(bbox_coordinate_to_ratio(bbox, height, width)),
							"attribute_values": attr,
							"attribute_type"  : attribute_type,
							"candidates"      : candidates,
							"answer"          : answer,
							"metadata"        : {
								"object"   : [name],
								"attribute": attr
							}
						},
					)

		return data_list


AttributeGeneratorList = [
	ExistsAttributeGenerator,
	# AttributeBBoxGenerator,
	# TypedAttributeBBoxGenerator
]
