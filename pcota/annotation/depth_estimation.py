import os.path

import requests
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from .utils import save_depth_res


def depth_estimation(
		image_list: list,
		model: str = "depth-anything/Depth-Anything-V2-Small-hf",
		device: str = "cpu",
		save_path: str = "./depth_pred_res/",
		**kwargs
):
	# load pipe
	pipe = pipeline(task="depth-estimation", model=model, device=device, **kwargs)

	# inference
	res = []
	for image in tqdm(image_list, desc="depth estimation"):
		if "http" in image:
			image_obj = Image.open(requests.get(image, stream=True).raw)
			image_name = image.split('/')[-1]
		else:
			image_obj = Image.open(image)
			image_name = os.path.basename(image)

		# inference
		depth = pipe(image_obj)['depth']
		img_path = save_depth_res((image_obj.height, image_obj.width),
								  depth, image_name, True, save_path)
		res.append({
			"depth_mask_id": os.path.basename(img_path)
		})

	return res

# if __name__ == "__main__":
#     image_list = ["/home/elpis_ubuntu/llm/InstructVerse/sample_data/images/demo.jpg"]
#     paths = depth_estimation(image_list, device="cuda:0")
