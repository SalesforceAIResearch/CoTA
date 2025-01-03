import requests
import torch
from PIL import Image

from tqdm import tqdm
from transformers import AutoProcessor, Owlv2ForObjectDetection


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
		device: str = "cpu",
		model_name: str = "google/owlv2-base-patch16-finetuned",
		threshold: float = 0.3,
		**kwargs
):
	processor = AutoProcessor.from_pretrained(model_name)
	model = Owlv2ForObjectDetection.from_pretrained(model_name)
	image_tag_list = _identify_object(image_list, device=device, **kwargs)
	_image_list = []
	for image_path_or_url in image_list:
		if "http" in image_path_or_url:
			image = Image.open(requests.get(image_path_or_url, stream=True).raw)
		else:
			image = Image.open(image_path_or_url)
		_image_list.append(image)

	inputs = processor(text=image_tag_list, images=_image_list, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**inputs)

	target_sizes = torch.Tensor([[image.size[::-1]] for image in _image_list]).reshape(len(_image_list), -1).tolist()
	# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
	results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

	return [{
		"det_scores": res["scores"].tolist(),
		"labels"    : [image_tag_list[i][l] for l in res["labels"].tolist()],
		"bboxes"    : torch.where(res["boxes"] > 0, res["boxes"], 0).to(torch.int32).tolist()
	} for i, res in enumerate(results)]

# if __name__ == "__main__":
# 	from utils import visualize_obj_det
#
# 	image_path_list = [
# 		"/home/elpis_ubuntu/llm/InstructVerse/sample_data/images/animals.png",
# 		"/home/elpis_ubuntu/llm/InstructVerse/sample_data/images/apples.jpg"
# 	]  # url or path
# 	res = object_detection(image_path_list, device="cuda:0")  # list[dict[labels, scores, bboxes]]
#
# 	visualize_obj_det(image_path_list[0], res[0]["bboxes"], res[0]["scores"], res[0]["labels"])
# 	visualize_obj_det(image_path_list[1], res[1]["bboxes"], res[1]["scores"], res[1]["labels"])
