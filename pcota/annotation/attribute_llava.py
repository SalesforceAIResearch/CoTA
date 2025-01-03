import json
import os
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import pandas as pd
from tqdm import tqdm

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file) -> Image:
	if image_file.startswith('http://') or image_file.startswith('https://'):
		response = requests.get(image_file)
		image = Image.open(BytesIO(response.content)).convert('RGB')
	else:
		image = Image.open(image_file).convert('RGB')
	return image


def get_attributes(
        image_list: list,
        obj_det_path: str,
        model_base: str = None,
        if_load_8bit: bool = False,
        if_load_4bit: bool = False,
        model_path: str = "jieyuz2/llava7b-attr3",
        device: str = "cuda",
        inp = "<image>\n{label}",
        conv_mode = None,
        temperature = 0.2,
        max_new_tokens = 512
    ):
	disable_torch_init()
	model_name = get_model_name_from_path(model_path)
	tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                           model_base, 
                                                                           model_name, 
                                                                           if_load_8bit, 
                                                                           if_load_4bit, 
                                                                           device=device)
	# Model
	if "llama-2" in model_name.lower():
		conv_mode = "llava_llama_2"
	elif "mistral" in model_name.lower():
		conv_mode = "mistral_instruct"
	elif "v1.6-34b" in model_name.lower():
		conv_mode = "chatml_direct"
	elif "v1" in model_name.lower():
		conv_mode = "llava_v1"
	elif "mpt" in model_name.lower():
		conv_mode = "mpt"
	else:
		conv_mode = "llava_v0"

	if conv_mode is not None and conv_mode != conv_mode:
		print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, conv_mode, conv_mode))
	else:
		conv_mode = conv_mode

	obj_det_res = json.load(open(obj_det_path, 'r'))
	results = []
	for image_path in tqdm(image_list, desc="Generating attributes"):
		image_id = image_path.split('/')[-1].split('.')[0]
		image_obj_labels = obj_det_res[image_id]['annotation']['labels']
		bboxes = obj_det_res[image_id]['annotation']['bboxes']
		image = load_image(image_path)
		# Similar operation in model_worker.py

		img_res = []
		for label, bbox in zip(image_obj_labels, bboxes):
			crop_img = image.crop(bbox)
			crop_img_size = crop_img.size
			crop_image_tensor = process_images([crop_img], image_processor, model.config)
			crop_image_tensor = crop_image_tensor.to(model.device, dtype=torch.float16)
   
			conv = conv_templates[conv_mode].copy()
			conv.append_message(conv.roles[0], inp.format(label=label))
			conv.append_message(conv.roles[1], None)
			prompt = conv.get_prompt()

			input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
			streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
			
			with torch.inference_mode():
				output_ids = model.generate(
					input_ids,
					images=crop_image_tensor,
					image_sizes=[crop_img_size],
					do_sample=True if temperature > 0 else False,
					temperature=temperature,
					max_new_tokens=max_new_tokens,
					streamer=streamer,
					use_cache=True)
			outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
			img_res.append([t.strip() for t in outputs.split(',')])
		results.append(img_res)

	return [{
		"attributes": res
    } for res in results]


# if __name__ == "__main__":
# 	image_list = ["/linxindisk/linxin/llm/InstructVerse/sample_data/images/demo.jpg"]
# 	preds = get_attributes(image_list, obj_det_path="/linxindisk/linxin/llm/InstructVerse/sample_data/object-detection-annotations.json")