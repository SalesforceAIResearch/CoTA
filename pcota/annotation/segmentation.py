import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from .utils import get_image, prepare_sam_model, save_sep_mask, save_mask_as_png

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    import warnings
    warnings.warn("segment_anything is not installed. Annotation functions can not be used.")



def sam_mask_prediction(
        image_list: list[str],
        path_to_checkpoint: str = "sam2_hiera_large.pt",
        save_path: str = "./seg_pred_res/",
        device: str = "cuda",
        sam2_version: str = "072824",
        sam2_config_path: str = "sam2_hiera_l.yaml",
        input_points: Optional[List[np.ndarray]] = None,
        input_labels: Optional[List[np.ndarray]] = None,
        bboxes: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        multimask_output: Optional[bool] = False,
        save_png: bool = False,
        **kwargs
):

    prepare_sam_model(path_to_checkpoint, version=sam2_version)
    predictor = SAM2ImagePredictor(build_sam2(sam2_config_path, path_to_checkpoint))

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        res = []
        for image_path_or_url, image_bboxes, image_label in tqdm(zip(image_list, bboxes, labels), desc="segmentation prediction"):
            raw_image = np.array(get_image(image_path_or_url))
            predictor.set_image(raw_image)

            # If multimask_output been set to True, it will return three masks with three scores that represent their quality.
            # Output format: Tuple[masks, scores, logits]
            masks = []
            for bbox in image_bboxes:
                mask = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=np.array(bbox),
                    multimask_output=multimask_output,
                    **kwargs
                )[0]
                masks.append({
                    "segmentation": mask[0],
                    "bbox": bbox
                })
            if "http" in image_path_or_url:
                image_name = image_path_or_url.split('/')[-1][:os.path.basename(image_path_or_url).rfind(".")]
            else:
                image_name = os.path.basename(image_path_or_url)[:os.path.basename(image_path_or_url).rfind(".")]
                
            whole_mask_path, _ = save_sep_mask(masks, save_path, image_name)
            
            if save_png:
                save_mask_as_png(masks, f"{save_path}/{image_name}_mask.png")
            
            res.append({
                "seg_mask_id": whole_mask_path.split('/')[-1],
                "bboxes": image_bboxes,
                "labels": image_label
            })
    return res


def sam_mask_generation(
        image_list: list[str],
        path_to_checkpoint: str = "sam2_hiera_large.pt",
        save_path: str = "./seg_pred_res",
        device: str = "cuda",
        sam2_version: str = "072824",
        sam2_config_path: str = "sam2_hiera_l.yaml",
        save_png: bool = False
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    prepare_sam_model(path_to_checkpoint, version=sam2_version)
    mask_generator = SAM2AutomaticMaskGenerator(build_sam2(sam2_config_path, path_to_checkpoint))

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        res = []
        for image_path_or_url in tqdm(image_list, desc="segmentation generation"):
            if "http" in image_path_or_url:
                image_name = image_path_or_url.split('/')[-1][:os.path.basename(image_path_or_url).rfind(".")]
            else:
                image_name = os.path.basename(image_path_or_url)[:os.path.basename(image_path_or_url).rfind(".")]
            raw_image = np.array(get_image(image_path_or_url))
            masks = mask_generator.generate(raw_image)
            # save_mask_as_png(masks, save_path + f"{image_name}_mask.png")  # whole mask
            whole_mask_path, bboxes = save_sep_mask(masks, save_path, image_name)
            
            if save_png:
                save_mask_as_png(masks, f"{save_path}/{image_name}_mask_all.png")
                
            res.append({
                "seg_mask_id": whole_mask_path.split('/')[-1],
                "bboxes": [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in bboxes]
            })

        return res

# if __name__ == "__main__":
#     from utils import visualize_obj_det
#     image_list = ["/linxindisk/linxin/llm/InstructVerse/sample_data/images/demo.jpg"]
#     preds = sam_mask_generation(image_list, device="cuda", save_png=True)
#     box_preds = sam_mask_prediction(image_list, bboxes=[[preds[0]['bboxes'][0]]], labels=[["None" for _ in range(len(preds[0]['bboxes']))]], device="cuda", save_png=True)
#     visualize_obj_det(image_list[0], preds[0]['bboxes'], [0 for _ in range(len(preds[0]['bboxes']))], ["None" for _ in range(len(preds[0]['bboxes']))])
