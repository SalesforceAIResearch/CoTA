from collections import defaultdict
from typing import List

import inflect
from tqdm import tqdm

from ..template import get_qa_template
from ..utils import *
from ...base import BaseMultiGenerator
from ...dataset import JointDataset


class CountObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "how many {object} in these images?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("CountObjectMultiGenerator")
	inflect_engine = inflect.engine()
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, number_mode="random", **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.number_mode = number_mode

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data)
			if check_object_for_counting_task(target_obj) and len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					cnt = 0
					all_bboxes = []
					for i in selected:
						num_bboxes = len(self.dataset.annotations[i].bboxes)
						bboxes = [self.dataset.annotations[i].bboxes[j] for j in range(num_bboxes) if self.dataset.annotations[i].labels[j] == target_obj]
						i_cnt = self.dataset.annotations[i].labels.count(target_obj)
						cnt += i_cnt
						assert len(bboxes) == i_cnt
						all_bboxes.append(bboxes)
					assert cnt > 0

					if cnt == 1:
						candidates = [1, 2, 3, 4]
					else:
						candidates = [cnt, cnt - 1, cnt + 1, cnt + 2]
					candidates = get_cnt_word(candidates, self.rng, self.inflect_engine, self.number_mode)
					answer = candidates[0]

					outputs = []
					new_image_paths = []
					for i, bboxes in enumerate(all_bboxes):
						scores = [round(self.rng.uniform(0.50, 1.00), 2) for _ in bboxes]
						di = selected[i] 
						annotation = self.dataset.annotations[di]
						height, width = self.dataset.annotations[di].height, self.dataset.annotations[di].width
						bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
						output = {"image": f"image-{i}: <image>", "regions": [{"label": target_obj, "bbox": bbox, "score": scores[j]} for j, bbox in enumerate(bboxes)]}
      
						tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [target_obj] * len(bboxes))
						ann_dir = os.path.dirname(self.dataset.annotation_path)
						new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
						new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-{i}.jpg"
						# new_image_path = f"{self.__class__.__name__}/{new_image_filename}"
						new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
						new_image_paths.append(new_image_path)
						outputs.append(output)
						
					data_list += self.make_one_data(
						{
							"data_path" : [self.dataset.data_paths[di] for di in selected],
							"object"    : target_obj,
							"candidates": candidates,
							"answer"    : answer,
							"metadata"  : {
								"object": [target_obj],
							},
							"tools"		: [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [target_obj]}} for i, di in enumerate(selected)],
							"outputs"	: outputs,
							"new_images": new_image_paths
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class CountAttributeObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "how many {attribute} {object} in these images?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("CountAttributeObjectMultiGenerator")
	inflect_engine = inflect.engine()
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, number_mode="random", **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		self.number_mode = number_mode

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data)
			if check_object_for_counting_task(target_obj) and len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					attribute_to_cnt = defaultdict(lambda: 0)
					all_bboxes = []
					for i, di in enumerate(selected):
						name_to_bboxes = defaultdict(list)
						annotation = self.dataset.annotations[di]
						for obj, attrs, bbox in zip(annotation.labels, annotation.attributes, annotation.bboxes):
							if obj == target_obj:
								for attr in attrs:
									attribute_to_cnt[attr] += 1
									name_to_bboxes[f"{attr} {target_obj}"].append(bbox)
						all_bboxes.append(name_to_bboxes)

					candidate_attrs = [attr for attr, cnt in attribute_to_cnt.items() if cnt > 1]
					if len(candidate_attrs) > 0:
						attribute = self.rng.choice(candidate_attrs)
						cnt = attribute_to_cnt[attribute]
						if cnt == 1:
							candidates = [1, 2, 3, 4]
						else:
							candidates = [cnt, cnt - 1, cnt + 1, cnt + 2]
						candidates = get_cnt_word(candidates, self.rng, self.inflect_engine, self.number_mode)
						answer = candidates[0]
						outputs = []
						name = f"{attribute} {target_obj}"
						new_image_paths = []
						for i, name2bboxes in enumerate(all_bboxes):
							bboxes = name2bboxes[name]
							scores = [round(self.rng.uniform(0.50, 1.00), 2) for _ in bboxes]
							di = selected[i] 
							height, width = self.dataset.annotations[di].height, self.dataset.annotations[di].width
							bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
							output = {"image": f"image-{i}: <image>", "regions": [{"label": name, "bbox": bbox, "score": scores[j]} for j, bbox in enumerate(bboxes)]}

							annotation = self.dataset.annotations[di]
							tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [name] * len(bboxes))
							ann_dir = os.path.dirname(self.dataset.annotation_path)
							new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
							new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-{i}.jpg"
							new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
							new_image_paths.append(new_image_path)
		
							outputs.append(output)
		
						data_list += self.make_one_data(
							{
								"data_path" : [self.dataset.data_paths[di] for di in selected],
								"object"    : target_obj,
								"attribute" : attribute,
								"candidates": candidates,
								"answer"    : answer,
								"metadata"  : {
									"object"   : [target_obj],
									"attribute": [attribute]
								},
								"tools"		: [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [name]}} for i, di in enumerate(selected)],
								"outputs"	: outputs,
								"new_images": new_image_paths
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


class CommonObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "What are common {object} in these images?",
	"response": "{objects}"
	"""

	qa_templates = get_qa_template("CommonObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data)
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					objects = set(self.dataset.annotations[selected[0]].labels)
					all_objects = set(objects)
					for i in selected[1:]:
						labels = set(self.dataset.annotations[i].labels)
						objects &= labels
						all_objects |= labels
					remaining_objects = all_objects - objects
					candidates = list(remaining_objects)
					if len(candidates) > 0:
						candidates = list(self.rng.choice(candidates, min(self.n_choice - 1, len(candidates)), replace=False))
						candidates = [target_obj] + candidates
						objects = list(objects)
						data_list += self.make_one_data(
							{
								"data_path" : [self.dataset.data_paths[di] for di in selected],
								"objects"   : objects,
								"candidates": candidates,
								"answer"    : target_obj,
								"metadata"  : {
									"object": objects,
								},
								"tools"		: [{"name": "GetObjects", "arguments": {"image": f"image-{i}"}} for i, di in enumerate(selected)],
								"outputs"	: [{"objects": list(set(self.dataset.annotations[i].labels))} for i in selected],
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


class CommonAttributeMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "What is the common attribute of {object} in these images?",
	"response": "{attributes}"
	"""

	qa_templates = get_qa_template("CommonAttributeMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data)
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					attribute_to_data = defaultdict(set)
					all_attributes = []
					for i, di in enumerate(selected):
						i_attributes = []
						annotation = self.dataset.annotations[di]
						for obj, attrs in zip(annotation.labels, annotation.attributes):
							if obj == target_obj:
								for attr in attrs:
									attribute_to_data[attr].add(i)
								i_attributes += attrs
						all_attributes.append(list(set(i_attributes)))

					attributes = [attr for attr, d in attribute_to_data.items() if len(d) == self.n_data]
					candidates = [a for a in attribute_to_data if a not in attributes]
						
					if len(attributes) > 0 and len(candidates) > 0:
						answer = self.rng.choice(attributes)
						if len(candidates) < self.n_choice - 1:
							candidates += [answer]
						else:
							candidates = self.rng.choice(candidates, self.n_choice - 1, replace=False).tolist() + [answer]
						data_list += self.make_one_data(
							{
								"data_path" : [self.dataset.data_paths[di] for di in selected],
								"object"    : target_obj,
								"attributes": attributes,
								"candidates": candidates,
								"answer"    : answer,
								"metadata"  : {
									"object"   : [target_obj],
									"attribute": attributes
								},
								"tools"		: [{"name": "GetAttributesOfObject", "arguments": {"image": f"image-{i}", "object": target_obj}} for i, di in enumerate(selected)],
								"outputs"	: [{"attributes": attributes} for attributes in all_attributes],
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


MultiAggregateGeneratorList = [
	# CommonObjectMultiGenerator,
	# CommonAttributeMultiGenerator,
	CountObjectMultiGenerator,
	CountAttributeObjectMultiGenerator
]
