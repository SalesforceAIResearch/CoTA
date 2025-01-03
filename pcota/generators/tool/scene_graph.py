from abc import abstractmethod
from typing import List

import inflect

from ..attribute_classifier import AttributeClassifier
# from ..scene_graph_caption import get_sg_desc
from .scene_graph_qa import generate_attribute_qa, generate_object_qa, generate_relation_qa
from ..utils import *
from ...base import BaseGenerator
from ...dataset import SceneGraph, BoundBoxes


class SceneGraphCaptionGenerator(BaseGenerator):
	des_templates = [
		{}
	]
	inflect_engine = inflect.engine()
	metadata: bool = False

	def _generate(self, annotation: SceneGraph) -> List:
		descriptions = []
		subgraphs = annotation.decompose()
		singular_node = []
		for subgraph in subgraphs:
			if len(subgraph) == 1:
				singular_node.append(subgraph)
			else:
				description = self._describe_subgraph(subgraph)
				descriptions.append(description)
		if len(singular_node):
			descriptions = [self._describe_nodes(singular_node)] + descriptions
		descriptions = [d.capitalize() for d in descriptions]
		description = '\n\n'.join(descriptions)
		return [
			{
				"description": description,
				"metadata"   : {}
			}
		]

	def _describe_nodes(self, nodes):
		labels = []
		for node in nodes:
			attr = node.attributes[0]
			label = node.labels[0]
			if len(attr):
				attr = make_and_description(attr, self.rng)
				labels.append(f'{attr} {label}')
			else:
				labels.append(label)
		be = 'are'
		if len(labels) == 1 and not self.inflect_engine.singular_noun(labels[0]):
			be = 'is'
		d = f'there {be} {make_and_description(labels, self.rng)}.'
		return d

	def _describe_subgraph(self, subgraph: SceneGraph) -> List:
		subgraph = subgraph.single_edge_scene_graph(self.rng)
		G = subgraph.graph.copy()
		return get_sg_desc(G, self.rng)


def graph_to_json(graph: SceneGraph):
	objects = {}
	for i, (name, attrs) in enumerate(zip(graph.labels, graph.attributes)):
		objects[i] = {
			"name"      : name,
			"attributes": attrs,
			"relations" : []
		}
	for (o1, rel, o2) in graph.relations:
		objects[o1]["relations"].append({
			"object": o2,
			"name"  : rel
		})
	return {
		"objects": objects
	}


class SceneGraphQAGenerator(BaseGenerator):
	qa_templates = [
		{
			"prompt"  : "{prompt}",
			"response": "{response}"
		}
	]
	metadata: bool = True

	def _generate(self, annotation: SceneGraph) -> List:
		qas = []
		subgraphs = annotation.decompose()
		for subgraph in subgraphs:
			if len(subgraph) > 1:
				qas += self._qa_subgraph(subgraph, annotation)
		return qas

	@abstractmethod
	def _qa_subgraph(self, subgraph: SceneGraph, annotation: SceneGraph) -> List:
		pass


class SceneGraphObjectQAGenerator(SceneGraphQAGenerator):
	def _qa_subgraph(self, subgraph: SceneGraph, annotation: SceneGraph) -> List:
		subgraph_json = graph_to_json(subgraph)
		qas = []
		for q, a, attr, ref_obj_attr_names in generate_object_qa(subgraph_json):
			answer = self.rng.choice(a)
			candidates = list(set([i['name'] for i in subgraph_json['objects'].values() if i['name'] not in a]))
			candidates = [answer] + safe_sample(self.rng, candidates, self.n_choice - 1)

			attr = attr.replace("object", "").strip()
			target_objects = [f"{attr} {obj}" if len(attr) > 0 else obj for obj in candidates]
			
			if len(ref_obj_attr_names) == 0:
				all_objects = candidates
				all_attributed_objects = target_objects
				ref_obj = None
			else:
				assert len(ref_obj_attr_names) == 1
				ref_obj = ref_obj_attr_names[0]
				all_attributed_objects = target_objects + [ref_obj]
				ref_attr, ref_obj = ref_obj.split(" ")[0], ref_obj.split(" ")[-1]
				all_objects = candidates + [ref_obj]
				

			tools = [{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": all_attributed_objects}}]
			all_regions = []
			flatten_bboxes, flatten_objects, flatten_scores = [], [], []
			for i, obj in enumerate(all_objects):
				attr_obj = all_attributed_objects[i]
				if obj == answer:
					bboxes = [bbox for j, bbox in enumerate(annotation.bboxes) if annotation.labels[j] == obj and attr in annotation.attributes[j]]
					# print("answer bbox:", len(bboxes))
				elif obj == ref_obj:
					bboxes = [bbox for j, bbox in enumerate(annotation.bboxes) if annotation.labels[j] == obj]
					# print("ref object:", len(bboxes))
				bboxes = [list(bbox_coordinate_to_ratio(bbox, annotation.height, annotation.width)) for bbox in bboxes]
				scores = [round(self.rng.uniform(0.50, 1.00), 2) for _ in bboxes]
				regions = [{"label": attr_obj, "bbox": bbox, "score": score} for bbox, score in zip(bboxes, scores)]
				all_regions += regions
				
				flatten_bboxes += bboxes
				flatten_objects += [attr_obj] * len(bboxes)
				flatten_scores += scores
			outputs = [{"region": all_regions}]
			# tagging and saving the tagged image
			tagged_image = annotate_image_with_bboxes(annotation.data_path, flatten_bboxes, flatten_scores, flatten_objects)
			ann_dir = os.path.dirname(self.dataset.annotation_path)
			new_images_dir = os.path.join(ann_dir, "new_images", self.data_version, self.__class__.__name__)
			new_image_filename = f"{get_filename_without_extension(annotation.data_path)}-{len(qas)}-1.jpg"
			new_image_path = save_image_to_directory(tagged_image, new_images_dir, new_image_filename)
   
			qas += self.make_one_data(
				{
					"prompt"    : q,
					"response"  : a,
					"type"      : "object",
					"candidates": candidates,
					"answer"    : answer,
					"metadata"  : {
						"object": a
					},
					"tools"		: tools,
					"outputs"	: outputs,
					"new_images" : [new_image_path]
				}
			)
		return qas


class SceneGraphRelationQAGenerator(SceneGraphQAGenerator):
	def __init__(self, dataset, **kwargs):
		super().__init__(dataset=dataset, **kwargs)
		relations = set()
		for annotation in dataset.annotations:
			for _, rel, _ in annotation.relations:
				relations.add(rel)
		self.relations = list(relations)

	def sample_relations(self, rng, n=1, exclude=[]):
		return safe_sample(rng, self.relations, n, exclude)

	def _qa_subgraph(self, subgraph: SceneGraph) -> List:
		subgraph_json = graph_to_json(subgraph)
		qas = []
		for q, a in generate_relation_qa(subgraph_json):
			answer = self.rng.choice(a)
			candidates = self.sample_relations(self.rng, n=self.n_choice, exclude=a)
			candidates = [answer] + candidates
			qas += self.make_one_data(
				{
					"prompt"    : q,
					"response"  : a,
					"type"      : "relation",
					"candidates": candidates,
					"answer"    : answer,
					"metadata"  : {
						"relation": a
					}
				}
			)
		return qas


class SceneGraphAttributeQAGenerator(SceneGraphQAGenerator):
	attribute_classifier = AttributeClassifier()

	def _qa_subgraph(self, subgraph: SceneGraph) -> List:
		subgraph_json = graph_to_json(subgraph)
		qas = []
		for q, attribute_type, a in generate_attribute_qa(subgraph_json):
			answer = self.rng.choice(a)
			candidates = self.attribute_classifier.sample_attribute_from_category(attribute_type, self.rng, n=self.n_choice - 1, exclude=a)
			candidates = [answer] + candidates
			qas += self.make_one_data(
				{
					"prompt"        : q,
					"response"      : a,
					"type"          : "attribute",
					"attribute_type": attribute_type,
					"candidates"    : candidates,
					"answer"        : answer,
					"metadata"      : {
						"attribute": a
					}
				}
			)
		return qas


SceneGraphGeneratorList = [
	# SceneGraphCaptionGenerator,
	SceneGraphObjectQAGenerator,
	# SceneGraphRelationQAGenerator,
	# SceneGraphAttributeQAGenerator
]
