import torch
import os
import re
import json
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, CLIPImageProcessor
from .osprey.model.language_model.osprey_llama import OspreyLlamaForCausalLM
from .osprey.mm_utils import tokenizer_image_token
from .osprey.conversation import conv_templates, SeparatorStyle
from .osprey.constants import IMAGE_TOKEN_INDEX
from .osprey.train.train import DataArguments
from functools import partial


data_args = DataArguments()
data_args.mm_use_im_start_end = False
data_args.is_multimodal = True


class Osprey:
    def __init__(self,
                 model_path: str,
                 clip_path: str,
                 device: str='cuda',
                 num_of_seg_per_instance: int=2,
                 init_qs_prompt: str="what is the relationship between "):
        
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        
        model_path = os.path.expanduser(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.device = device
        if device == 'cpu':
            self.model = OspreyLlamaForCausalLM.from_pretrained(
                model_path,
                mm_vision_tower=clip_path).to(device)
            
            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device)
        else:
            self.model = OspreyLlamaForCausalLM.from_pretrained(
                model_path,
                mm_vision_tower=clip_path,
                torch_dtype=torch.bfloat16).to(device)
            
            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(dtype=torch.float16, device=device)
        
        self.image_processor = CLIPImageProcessor(do_resize=True, 
                                                  size={"shortest_edge":512}, 
                                                  resample=3,
                                                  do_center_crop=True, 
                                                  crop_size={"height": 512, "width": 512},
                                                  do_rescale=True, rescale_factor=0.00392156862745098, 
                                                  do_normalize=True, 
                                                  image_mean=[0.48145466, 0.4578275, 0.40821073],
                                                  image_std=[0.26862954, 0.26130258, 0.27577711], 
                                                  do_convert_rgb=True)
        
        spi_tokens = ['<mask>', '<pos>']
        self.tokenizer.add_tokens(spi_tokens, special_tokens=True)
        
        for m in self.model.modules():
            m.tokenizer = self.tokenizer
        
        rel_prompt = ''
        for i in range(num_of_seg_per_instance):
            rel_prompt = rel_prompt +  f'region{i+1} <mask><pos>' + ','
        
        begin_str = """<image>\n\nThis provides an overview of the picture.\n"""
        rel_prompt = f'There are {rel_prompt[:-1]} in the image, '
        qs = begin_str + rel_prompt + init_qs_prompt + ' and '.join([f'region{i+1}' for i in range(num_of_seg_per_instance)]) + '?'
        
        conv = conv_templates['osprey_v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        self.input_ids = tokenizer_image_token(prompt, 
                                               self.tokenizer, 
                                               IMAGE_TOKEN_INDEX, 
                                               return_tensors='pt').unsqueeze(0).to(self.model.device)
        self.stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        
        
    def cate_grounding(self, 
                       output_list: list, 
                       categlories_list_path: str,
                       batch_size = 256,
                       st_model_name: str = "all-mpnet-base-v2",
                       top_k: int = 1):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(st_model_name).to(self.device)
        
        cate_dict = json.load(open(categlories_list_path, 'r'))
        cate_list = []
        for sub_cate_list in cate_dict.values():
            cate_list += sub_cate_list
        cate_embedding = model.encode(cate_list, batch_size = batch_size)  # len(cate_list), 768
        
        relation_list = [triplet[1] for triplet in output_list]
        relation_embedding = model.encode(relation_list, batch_size = batch_size)  # len(relation_list), 768
        
        selected_cate_idx = cosine_similarity(relation_embedding, cate_embedding).argsort(axis=1)[:, -top_k:]
        new_relation_list = []
        for i, idx_list in enumerate(selected_cate_idx):
            for idx in idx_list:
                new_relation_list.append((output_list[i][0], cate_list[idx], output_list[i][2]))
        return {
            "relations": new_relation_list
        }


    def relation_detection(self, 
                           image_list: list,
                           seg_annotation_path: str,
                           seg_dir_path: str):
        res = []
        seg_annotation = json.load(open(seg_annotation_path, 'r'))
        for image_path in tqdm(image_list, desc="Relation Detection: "):
            img_id = image_path.split('/')[-1].split('.')[0]
            img = Image.open(image_path)
            annotation = seg_annotation[img_id]['annotation']  # [n_bboxes, w, h]
            raw_masks = np.load(os.path.join(seg_dir_path, annotation['seg_mask_id']))
            image = self.image_processor.preprocess(img, 
                                                    do_center_crop=False,
                                                    return_tensors='pt')['pixel_values'][0]
            
            image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                    size=(512, 512),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0)
            img_relations = []
            for idx1, mask1 in enumerate(raw_masks):
                if idx1 == len(raw_masks) - 1:
                    break
                for idx2, mask2 in enumerate(raw_masks[idx1 + 1:, :, :]):
                    masks = torch.from_numpy(np.array([mask1, mask2])).to(self.device)
                    
                    with torch.inference_mode():
                        self.model.orig_forward = self.model.forward
                        if self.model.device.type != 'cpu':
                            self.model.forward = partial(self.model.orig_forward,
                                                         img_metas=[None],
                                                         masks=[masks.half()])
                            
                            output_ids = self.model.generate(self.input_ids,
                                                             images=image.unsqueeze(0).half().to(self.device),
                                                             do_sample=True,
                                                             temperature=0.2,
                                                             max_new_tokens=100,
                                                             use_cache=False,
                                                             num_beams=1)
                        else:
                            self.model.forward = partial(self.model.orig_forward,
                                                         img_metas=[None],
                                                         masks=[masks])
                            
                            output_ids = self.model.generate(self.input_ids,
                                                             images=image.unsqueeze(0).to(self.device),
                                                             do_sample=True,
                                                             temperature=0.2,
                                                             max_new_tokens=100,
                                                             use_cache=False,
                                                             num_beams=1)

                        self.model.forward = self.model.orig_forward

                    input_token_len = self.input_ids.shape[1]
                    n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
                    
                    if n_diff_input_output > 0:
                        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                    
                    outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:],
                                                          skip_special_tokens=True)[0]

                    outputs = outputs.strip()
                    if outputs.endswith(self.stop_str):
                        outputs = outputs[:-len(self.stop_str)]
                    
                    outputs = outputs.strip()
                    if ':' in outputs:
                        outputs = outputs.split(':')[1]

                    outputs_list = outputs.split(',')
                    outputs_list_final = []
                    for output in outputs_list:
                        if output not in outputs_list_final:
                            if output == '':
                                continue
                            relation = re.sub(r'region\d+', '', output)
                            region_number = re.findall(r'region(\d+)', output)[0]
                            
                            outputs_list_final.append(relation)
                            # the "idx" is in the order of bboxes and labels
                            if region_number == '2':
                                img_relations.append((idx1 + idx2 + 1, relation, idx1))  # label_idx2, relation, label_idx1
                            else:
                                img_relations.append((idx1, relation, idx1 + idx2 + 1))  # label_idx1, relation, label_idx2
                        else:
                            break
            res.append(img_relations)
        return [{
            "relations": res
        } for res in img_relations]
                    


# if __name__ == "__main__":
#     image_path_list = [
#         "/linxindisk/linxin/llm/InstructVerse/sample_data/images/fruit.jpg",
#     ]  # url or path
#     model = Osprey(
#         model_path='linxinso/osprey_relation',
#         clip_path='/linxindisk/linxin/llm/InstructVerse/osprey_checkpoint/multi_region_v5_gqa_cot_bs16/open_clip_pytorch_model.bin',
#         device='cpu'
#     )
#     res = model.relation_detection(
#         image_path_list,
#         seg_annotation_path='/linxindisk/linxin/llm/InstructVerse/sample_data/segmentation-annotations.json',
#         seg_dir_path='/linxindisk/linxin/llm/InstructVerse/sample_data/seg_masks'
#     )
#     grounded_res = model.cate_grounding(
#         output_list = res['relations'],
#         categlories_list_path = "/linxindisk/linxin/llm/InstructVerse/relations.json"
#     )