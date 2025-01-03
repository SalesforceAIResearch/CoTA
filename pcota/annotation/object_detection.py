import requests
import cv2
import supervision as sv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms



def _identify_object(
		image_list: list[str],
		image_size: int = 384,
		model_path_or_url: str = "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth?download=true",
		vit_size: str = "swin_l",
		device: str = "cpu"
):
	from ram.models import ram_plus
	from ram import get_transform, inference_ram_openset as inference
	transform = get_transform(image_size=image_size)

	# load model
	model = ram_plus(pretrained=model_path_or_url,
					 image_size=image_size,
					 vit=vit_size)
	model.eval()
	model = model.to(device)

	res = []
	for image_path_or_url in tqdm(image_list, desc="tag generation"):
		image = transform(Image.open(image_path_or_url)).unsqueeze(0).to(device)
		tags = inference(image, model)
		res.append(tags.split(" | "))

	return res


def object_detection(
		image_list: list[str],
		model_path: str,
		model_config_path: str,
		device: str = "cpu",
		score_thres: float = 0.05,
		max_num_boxes: int = 100,
		iou_thres: float = 0.5,
		**kwargs
):
	image_tag_list = _identify_object(image_list, device=device, **kwargs)
	_image_list = []
	for image_path_or_url in image_list:
		if "http" in image_path_or_url:
			image = Image.open(requests.get(image_path_or_url, stream=True).raw)
		else:
			image = Image.open(image_path_or_url)
		_image_list.append(image)

	cfg = Config.fromfile(model_config_path)
	cfg.work_dir = "objdet_cache"
	cfg.load_from = model_path
	runner = Runner.from_cfg(cfg)
	runner.call_hook("before_run")
	runner.load_or_resume()
	pipeline = cfg.test_dataloader.dataset.pipeline
	runner.pipeline = Compose(pipeline)

	# run model evaluation
	runner.model.eval()
	
	results = []
	for idx, image_path in enumerate(image_list):
		texts = [[t.strip()] for t in image_tag_list[idx]] + [[" "]]
		data_info = runner.pipeline(dict(img_id=0, img_path=image_path, texts=texts))

		data_batch = dict(
			inputs=data_info["inputs"].unsqueeze(0),
			data_samples=[data_info["data_samples"]],
		)

		with autocast(enabled=False), torch.no_grad():
			output = runner.model.test_step(data_batch)[0]
			runner.model.class_names = texts
			pred_instances = output.pred_instances

		# nms
		keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=iou_thres)
		pred_instances = pred_instances[keep_idxs]
		pred_instances = pred_instances[pred_instances.scores.float() > score_thres]

		if len(pred_instances.scores) > max_num_boxes:
			indices = pred_instances.scores.float().topk(max_num_boxes)[1]
			pred_instances = pred_instances[indices]
		output.pred_instances = pred_instances

		# predictions
		results.append(pred_instances.cpu().numpy())

	return [{
		"det_scores": res["scores"].tolist(),
		"labels"    : [image_tag_list[i][l] for l in res["labels"].tolist()],
		"bboxes"    : res["bboxes"].astype(np.int32).tolist()
	} for i, res in enumerate(results)]

if __name__ == "__main__":
	from utils import visualize_obj_det

	image_path_list = [
		"/linxindisk/linxin/llm/InstructVerse/sample_data/images/animals.png",
		"/linxindisk/linxin/llm/InstructVerse/sample_data/images/apples.jpg"
	]  # url or path
	res = object_detection(
    	image_path_list, 
		model_path="yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth",
		model_config_path="yolo_configs/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py",
    	device="cuda:0"
    )  # list[dict[labels, det_scores, bboxes]]

	visualize_obj_det(image_path_list[0], res[0]["bboxes"], res[0]["det_scores"], res[0]["labels"])
	visualize_obj_det(image_path_list[1], res[1]["bboxes"], res[1]["det_scores"], res[1]["labels"])
