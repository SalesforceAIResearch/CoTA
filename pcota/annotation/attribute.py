import json
from itertools import chain

import diskcache
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

from ..generators.attribute_category import ATTRIBUTE_CATEGORY


class NgramCounter:
	def __init__(self, threshold=10, cache_path='.cache'):
		self.cache = {}
		self.cache_path = cache_path
		self.attributes = ATTRIBUTE_CATEGORY.copy()

		self.attributes.pop('activity')
		self.attributes.pop('size')
		self.attributes.pop('other')

		self.threshold = threshold

	def get_candidates(self, obj):
		if obj in self.cache:
			return self.cache[obj]

		cnt = {k: [self._request(a + ' ' + obj) for a in v] for k, v in self.attributes.items()}
		cnt_list = list(chain(*cnt.values()))
		threshold = np.mean(cnt_list)
		attributes = {k: [self.attributes[k][i] for i, c in enumerate(v) if c > threshold] for k, v in cnt.items()}

		print(f"{obj}: {np.mean(cnt_list)} +- {np.std(cnt_list)}")
		print(f"{obj}: {attributes}")

		self.cache[obj] = attributes
		return attributes

	def _request(self, query):
		with diskcache.Cache(self.cache_path, size_limit=100 * (2 ** 30)) as cache:
			cnt = cache.get(query, None)
			if cnt is None:
				payload = {
					'index'     : 'v4_piletrain_llama',
					'query_type': 'count',
					'query'     : query,
				}
				result = requests.post('https://api.infini-gram.io/', json=payload).json()
				cnt = result['count']
				cache.set(query, cnt)
		return cnt


attribute_prompt = [
	"a photo of {a}",
	"a photo of something {a}",
	"a photo of something that is {a}",
	"a cropped photo of {a}",
	"a cropped photo of something {a}",
	"a cropped photo of something that is {a}",
]

object_attribute_prompt = [
	"{a} {o}",
	"{o} {a}",
	"a photo of {a} {o}",
	"a photo of {o} {a}",
	"a cropped photo of {a} {o}",
	"a cropped photo of {o} {a}",
]


def get_object_attribute_features(obj, attributes, tokenizer, model, device):
	object_attribute_embedding = []
	for a in attributes:
		text = [prompt.format(a=a, o=obj) for prompt in object_attribute_prompt]
		input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
		text_features = model.encode_text(input_ids)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		text_features = text_features.mean(dim=0).cpu()
		object_attribute_embedding.append(text_features)
	return torch.vstack(object_attribute_embedding)


def get_attribute_embedding(attributes, tokenizer, model, device):
	attribute_embedding = []
	for a in attributes:
		text = [prompt.format(a=a) for prompt in attribute_prompt]
		input_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids.to(device)
		text_features = model.encode_text(input_ids)
		text_features /= text_features.norm(dim=-1, keepdim=True)
		text_features = text_features.mean(dim=0).cpu()
		attribute_embedding.append(text_features)
	return torch.vstack(attribute_embedding)


@torch.no_grad()
def get_attributes(
		image_list: list,
		bboxes_path: str,
		k: int = 3,
		thres: float = 0.2,
		model_name: str = "BAAI/EVA-CLIP-8B",
		device: str = "cuda"
):
	from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
	model = AutoModel.from_pretrained(
		model_name,
		torch_dtype=torch.float16,
		trust_remote_code=True).to(device).eval()
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

	ngram_counter = NgramCounter()

	bboxes_json = json.load(open(bboxes_path, 'r'))
	results = []
	with torch.no_grad(), torch.cuda.amp.autocast():
		text_feature_cache = {}
		for image_path in tqdm(image_list, desc="attributes"):
			image_id = image_path.split('/')[-1].split('.')[0]
			bboxes = bboxes_json[image_id]['annotation']['bboxes']
			labels = bboxes_json[image_id]['annotation']['labels']
			image = Image.open(image_path)

			image_bbox_attribute_list = []
			for label, bbox in zip(labels, bboxes):
				candidate_attributes = ngram_counter.get_candidates(label)
				if sum(map(len, candidate_attributes.values())) == 0:
					image_bbox_attribute_list.append([])
					continue

				if label in text_feature_cache:
					attribute_embedding, object_attribute_embedding, break_point, candidate_attributes_list = text_feature_cache[label]

				else:
					candidate_attributes_list = []
					break_point = []
					for k, v in candidate_attributes.items():
						if len(v):
							candidate_attributes_list.extend(v)
							break_point.append(len(candidate_attributes_list))
					attribute_embedding = get_attribute_embedding(candidate_attributes_list, tokenizer, model, device)
					object_attribute_embedding = get_object_attribute_features(label, candidate_attributes_list, tokenizer, model, device)
					text_feature_cache[label] = (attribute_embedding, object_attribute_embedding, break_point, candidate_attributes_list)

				attribute_embedding = attribute_embedding.to(device)
				object_attribute_embedding = object_attribute_embedding.to(device)

				crop_img = image.crop(bbox)
				input_pixels = processor(images=crop_img, return_tensors="pt", padding=True).pixel_values.to(device)

				image_features = model.encode_image(input_pixels)
				image_features /= image_features.norm(dim=-1, keepdim=True)
				scores1 = image_features @ attribute_embedding.T
				scores2 = image_features @ object_attribute_embedding.T
				scores = (scores1 + scores2) / 2

				scores = scores[0].cpu()
				start = 0
				selected_attribute_list = []
				for end in break_point:
					s = scores[start:end]
					if len(s) == 1:
						if s[0] > thres:
							selected_attribute_list.append(candidate_attributes_list[start])
					else:
						i = scores[start:end].argmax() + start
						if scores[i] > thres:
							selected_attribute_list.append(candidate_attributes_list[i])
					start = end
				image_bbox_attribute_list.append(selected_attribute_list)

				print(f"{label}: {selected_attribute_list}", scores.tolist())

			results.append(image_bbox_attribute_list)

	return [{
		"attributes": res,
	} for res in results]
