from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm

from ..template import get_qa_template
from ..utils import *
from ...base import BaseMultiGenerator
from ...dataset import JointDataset


class HasObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		total_number = len(self.dataset.annotations)
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			selected = tuple(self.rng.choice(total_number, self.n_data, replace=False, shuffle=False))
			if selected not in samples:
				samples.add(selected)

				object_to_data = defaultdict(list)
				all_bboxes = []
				selected = list(self.rng.permutation(selected))
				for i, di in enumerate(selected):
					annotation = self.dataset.annotations[di]
					obj_to_bboxes = defaultdict(list)
					for obj in set(annotation.labels):
						object_to_data[obj].append(i)
						bboxes = [bbox for j, bbox in enumerate(annotation.bboxes) if annotation.labels[j] == obj]
						obj_to_bboxes[obj] += bboxes
					all_bboxes.append(obj_to_bboxes)

				candidate_objs = [obj for obj, d in object_to_data.items() if len(d) == 1]
				if len(candidate_objs) > 0:
					obj = self.rng.choice(candidate_objs)
					d = object_to_data[obj]
					answer = d[0]
					if self.rng.choice([True, False]):
						tools = [{"name": "GetObjects", "arguments": {"image": f"image-{i}"}} for i, di in enumerate(selected)]
						outputs = [{"objects": list(set(self.dataset.annotations[di].labels))} for i, di in enumerate(selected)]
						new_image_paths = None
					else:
						tools = [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [obj]}} for i, di in enumerate(selected)]
						outputs, new_image_paths = [], []
						for i, di in enumerate(selected):
							if i == answer:
								bboxes = all_bboxes[i][obj]
								height, width = self.dataset.annotations[di].height, self.dataset.annotations[di].width
								bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
								scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in bboxes]

								output = {"image": f"image-{len(selected)+i}: <image>", "regions": [{"label": obj, "bbox": bbox, "score": scores[i]} for i, bbox in enumerate(bboxes)]}
							else:
								output = {"image": f"image-{len(selected)+i}: <image>", "regions": []}
								bboxes = []
								scores = []
							tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [obj] * len(bboxes))
							ann_dir = os.path.dirname(self.dataset.annotation_path)
							new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
							new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-{i}.jpg"
							# new_image_path = f"{self.__class__.__name__}/{new_image_filename}"
							new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
							new_image_paths.append(new_image_path)
							outputs.append(output)

					data_dict = {
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : obj,
							"answer_id": answer,
							"metadata" : {
								"object": [obj],
							},
							"tools": tools,
							"outputs": outputs,
						}
					if new_image_paths is not None:
						data_dict.update({"new_images": new_image_paths})
					data_list += self.make_one_data(data_dict)
					pbar.update(1)

		pbar.close()
		return data_list


class HasAttributedObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has {attribute} {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasAttributedObjectMultiGenerator")
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
					all_bboxes = []
					selected = list(self.rng.permutation(selected))
					for i, di in enumerate(selected):
						annotation = self.dataset.annotations[di]
						attribute_to_bboxes = defaultdict(list)
						for obj, attrs, bbox in zip(annotation.labels, annotation.attributes, annotation.bboxes):
							if obj == target_obj:
								for attr in attrs:
									attribute_to_data[attr].add(i)
									attribute_to_bboxes[attr].append(bbox)
						all_bboxes.append(attribute_to_bboxes)

					candidate_attrs = [attr for attr, d in attribute_to_data.items() if len(d) == 1]
					if len(candidate_attrs) > 0:
						attribute = self.rng.choice(candidate_attrs)
						d = list(attribute_to_data[attribute])
						name = f"{attribute} {target_obj}"
						answer = d[0]
						tools = [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [name]}} for i, di in enumerate(selected)]
						outputs = []
						new_images_paths = []
						for i, di in enumerate(selected):
							
							if i == answer:
								bboxes = all_bboxes[i][attribute]
								height, width = self.dataset.annotations[di].height, self.dataset.annotations[di].width
								bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
								scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in bboxes]
								output = {"image": f"image-{len(selected)+i}: <image>", "regions": [{"label": name, "bbox": bbox, "score": scores[i]} for i, bbox in enumerate(bboxes)]}
							else:
								bboxes = []
								scores = []
								output = {"image": f"image-{len(selected)+i}: <image>", "regions": []}
							outputs.append(output)
							tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [name] * len(bboxes))
							ann_dir = os.path.dirname(self.dataset.annotation_path)
							new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
							new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-{i}.jpg"
							# new_image_path = f"{self.__class__.__name__}/{new_image_filename}"
							new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
							new_images_paths.append(new_image_path)

						data_list += self.make_one_data(
							{
								"data_path": [self.dataset.data_paths[di] for di in selected],
								"object"   : target_obj,
								"attribute": attribute,
								"answer_id": answer,
								"metadata" : {
									"object"   : [target_obj],
									"attribute": [attribute]
								},
								"tools": tools,
								"outputs": outputs,
								"new_images": new_images_paths
							},
						)
						pbar.update(1)

		pbar.close()
		return data_list


class HasNotObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image does not have  {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, self.n_data - 1)
			d = self.dataset.sample_data_without_obj(self.rng, target_obj, 1)
			selected = d + selected
			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if (target_obj, selected) not in samples:
					samples.add((target_obj, selected))

					answer = selected.index(d[0])
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : target_obj,
							"answer_id": answer,
							"metadata" : {
								"object": [target_obj],
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasNotAttributedObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image does not have {attribute} {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotAttributedObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			target_obj, selected = self.dataset.sample_data_and_obj(self.rng, 2)
			selected = tuple(sorted(selected))
			if (target_obj, selected) not in samples:
				samples.add((target_obj, selected))

				attribute_to_data = defaultdict(set)
				for i, di in enumerate(selected):
					annotation = self.dataset.annotations[di]
					for obj, attrs in zip(annotation.labels, annotation.attributes):
						if obj == target_obj:
							for attr in attrs:
								attribute_to_data[attr].add(i)

				candidate_attrs = [attr for attr, d in attribute_to_data.items() if len(d) == 1]
				if len(candidate_attrs) > 0:
					attribute = self.rng.choice(candidate_attrs)
					d = list(attribute_to_data[attribute])
					answer = 1 - d[0]

					if self.n_data > 2:
						others = self.dataset.sample_data_without_obj(self.rng, target_obj, self.n_data - 2)
						selected = list(selected) + others

					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"object"   : target_obj,
							"attribute": attribute,
							"answer_id": answer,
							"metadata" : {
								"object"   : [target_obj],
								"attribute": [attribute]
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasRelationMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "In which image {object1} is {relation} {object2}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasRelationMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:

			objs, rel, d1 = self.dataset.sample_data_and_rel(self.rng, 1)
			d2 = self.dataset.sample_data_without_rel(self.rng, objs, rel, self.n_data - 1)
			selected = d1 + d2

			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if selected not in samples:
					samples.add(selected)
					selected = list(self.rng.permutation(selected))
					answer = selected.index(d1[0])
					tools, outputs = [], []

					for i, di in enumerate(selected):
						tool = {"name": "GetRelationshipsBetweenObjects", "arguments": {"image": f"image-{i}", "object1":  objs[0], "object2": objs[1]}}
						annotation = self.dataset.annotations[di]
						# agg_relations = defaultdict(list)
						relations = set()
						for o1, relation, o2 in annotation.relations:
							o1, o2 = annotation.labels[o1], annotation.labels[o2]
							if o1 == objs[0] and o2 == objs[1]:
								relations.add(relation)
						if i == answer:
							assert rel in relations
						else:
							assert rel not in relations
							# agg_relations[(o1, o2)].append(relation)
						output = {"relations": list(relations)}
						tools.append(tool)
						outputs.append(output)
							
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"relation" : rel,
							"object1"  : objs[0],
							"object2"  : objs[1],
							"answer_id": answer,
							"metadata" : {
								"relation": [rel],
								"object"  : list(objs),
							},
							"tools": tools,	
							"outputs": outputs
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasNotRelationMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "In which image {object1} is not {relation} {object2}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasNotRelationMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:

			objs, rel, d1 = self.dataset.sample_data_and_rel(self.rng, self.n_data - 1)
			d2 = self.dataset.sample_data_without_rel(self.rng, objs, rel, 1)
			selected = d1 + d2

			if len(selected) == self.n_data:
				selected = tuple(sorted(selected))
				if selected not in samples:
					samples.add(selected)

					answer = selected.index(d2[0])
					data_list += self.make_one_data(
						{
							"data_path": [self.dataset.data_paths[di] for di in selected],
							"relation" : rel,
							"object1"  : objs[0],
							"object2"  : objs[1],
							"answer_id": answer,
							"metadata" : {
								"relation": [rel],
								"object"  : list(objs),
							}
						},
					)
					pbar.update(1)

		pbar.close()
		return data_list


class HasMostObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has most {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasMostObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			res = self.dataset.sample_data_and_obj_diff_cnt(self.rng, self.n_data)
			if len(res):
				target_obj, l = res
				if check_object_for_counting_task(target_obj):
					l = list(self.rng.permutation(l))
					selected, cnt = zip(*l)
					if len(selected) == self.n_data:
						selected_ = tuple(sorted(selected))
						if selected_ not in samples:
							samples.add(selected_)
							
							answer = int(np.argmax(cnt))
							tools = [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [target_obj]}} for i, di in enumerate(selected)]
							outputs, new_image_paths = [], []
							for i, di in enumerate(selected):
								annotation = self.dataset.annotations[di]
								bboxes = [bbox for j, bbox in enumerate(annotation.bboxes) if annotation.labels[j] == target_obj]
								height, width = annotation.height, annotation.width
								bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
								assert len(bboxes) == cnt[i]
								scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in bboxes]
								output = {"image": f"image-{len(selected)+i}: <image>", "regions": [{"label": target_obj, "bbox": bbox, "score": scores[i]} for i, bbox in enumerate(bboxes)]}
								outputs.append(output)

								tagged_image = annotate_image_with_bboxes(annotation.data_path, bboxes, scores, [target_obj] * len(bboxes))
								ann_dir = os.path.dirname(self.dataset.annotation_path)
								new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
								new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(data_list)}-{i}.jpg"
								# new_image_path = f"{self.__class__.__name__}/{new_image_filename}"
								new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
								new_image_paths.append(new_image_path)
       
							data_list += self.make_one_data(
								{
									"data_path": [self.dataset.data_paths[di] for di in selected],
									"object"   : target_obj,
									"answer_id": answer,
									"metadata" : {
										"object": [target_obj],
									},
									"tools": tools,
									"outputs": outputs,
									"new_images": new_image_paths
								},
							)
							pbar.update(1)

		pbar.close()
		return data_list


class HasLeastObjectMultiGenerator(BaseMultiGenerator):
	"""
	"prompt"  : "which image has least {object}?",
	"response": "{answer}"
	"""

	qa_templates = get_qa_template("HasLeastObjectMultiGenerator")
	metadata: bool = True
	dataset: JointDataset

	def __init__(self, dataset: JointDataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)

	def _generate(self, n_sample: int) -> List:
		data_list = []

		samples = set()
		pbar = tqdm(total=n_sample, desc=self.__class__.__name__, position=0, leave=True)
		while len(data_list) < n_sample:
			res = self.dataset.sample_data_and_obj_diff_cnt(self.rng, self.n_data)
			if len(res):
				target_obj, l = res
				if check_object_for_counting_task(target_obj):
					l = list(self.rng.permutation(l))
					selected, cnt = zip(*l)
					if len(selected) == self.n_data:
						selected_ = tuple(sorted(selected))
						if selected_ not in samples:
							samples.add(selected_)

							answer = int(np.argmin(cnt))
							tools = [{"name": "LocalizeObjects", "arguments": {"image": f"image-{i}", "objects": [target_obj]}} for i, di in enumerate(selected)]
							outputs, new_image_paths = [], []
							for i, di in enumerate(selected):
								annotation = self.dataset.annotations[di]
								bboxes = [bbox for j, bbox in enumerate(annotation.bboxes) if annotation.labels[j] == target_obj]
								height, width = annotation.height, annotation.width
								bboxes = [list(bbox_coordinate_to_ratio(bbox, height, width)) for bbox in bboxes]
								assert len(bboxes) == cnt[i]
								scores = [float(round(self.rng.uniform(0.50, 1.00), 2)) for _ in bboxes]
								output = {"image": f"image-{len(selected)+i}: <image>", "regions": [{"label": target_obj, "bbox": bbox, "score": scores[j]} for j, bbox in enumerate(bboxes)]}
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
									"data_path": [self.dataset.data_paths[di] for di in selected],
									"object"   : target_obj,
									"answer_id": answer,
									"metadata" : {
										"object": [target_obj],
									},
									"tools": tools,
									"outputs": outputs,
									"new_images": new_image_paths
								},
							)
							pbar.update(1)

		pbar.close()
		return data_list


MultiSelectGeneratorList = [
	HasRelationMultiGenerator,
	# HasNotRelationMultiGenerator,
	HasObjectMultiGenerator,
	# HasNotObjectMultiGenerator,
	HasAttributedObjectMultiGenerator,
	# HasNotAttributedObjectMultiGenerator,
	HasMostObjectMultiGenerator,
	HasLeastObjectMultiGenerator
]
