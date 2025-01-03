import os
import re
import pickle
import json
import PIL
from PIL import Image
import numpy as np
from taco.config import *

def reformat_text(text):
    pattern = r'#\s*STEP\s*\d?\s*#\:?' # to match one of these: `# STEP 1 #:`, `# STEP #:`, `#STEP 1#:`, `#STEP#:`, or `#STEP#`
    match = re.search(pattern, text)
    if match:
        end_index = match.end()
        text = text[end_index:]
    text = text.replace("```", "")
    text = text.replace("json", "")
    text = text.replace("\n", "")
    return text
    
def convert_openai_data_to_llava_format(example, id, metadata, image_base_path):
    llava_example = {'id': id, 'task_instruction_id': 0, 'conversations': [], 'metadata': metadata, 'image': [], 'choice_list': None}
    
    # try:
    all_msgs_path = example['all_messages']
    all_msgs = pickle.load(open(all_msgs_path, 'rb'))
    msgs = all_msgs['user_agent']
    convos = []
    images = []
    for i, msg in enumerate(msgs):
        convo = {}
        if msg['role'] == 'user':
            convo['from'] = 'human'
        elif msg['role'] == 'assistant':
            convo['from'] = 'gpt'
        else:
            raise NotImplementedError
        
        convo_content = ""
        for j, content in enumerate(msg['content']):
            if content['type'] == "text":
                convo_content += content['text'] + '\n'
            elif content['type'] == "image_url":
                pil_image = content['image_url']['url']
                convo_content += '<image>\n'
                image_path = os.path.join(image_base_path, metadata['dataset'], f"{id}-{i}-{j}.jpg")
                pil_image.save(image_path)
                images.append(image_path)
                    
            else:
                raise NotImplementedError
        if convo['from'] == 'gpt':
            convo_content = reformat_text(convo_content)
            try:
                parsed_content = json.loads(convo_content)
            except Exception as e:
                print("CONVO:", convo_content)
                print(e)
                return None
            
        convo['value'] = convo_content
        convos.append(convo)
    llava_example['conversations'] = convos
    llava_example['image'] = images
    
    return llava_example


def remove_input_image_path(text):
    pattern = r"Image(?: [0-9])? path: .*\n"
    match = re.search(pattern, text)
    if match:
        end_index = match.end()
        start_index = match.start()
        text = text[:start_index] + text[end_index:]
    return text

def rewrite_image_argument(text, new_content):
    # The regex pattern
    pattern = r'["\']image["\']: ["\'](.*?)["\']'

    # Perform the search
    match = re.search(pattern, text)
    # Check if a match is found and get the start and end index of the content
    if match:
        old_content = match.group(1)
        print("Content found:", old_content)
        text = text.replace(old_content, new_content)
        print("New text:", text)
        return text
    else:
        return text
    

def get_files(directory):
    items = os.listdir(directory)
    # Filter out only files
    files = [os.path.join(directory, item) for item in items if os.path.isfile(os.path.join(directory, item))]
    return files

def load_msgs_in_example(example):
    if isinstance(example['all_messages'], list):
        return example['all_messages']
    all_msgs_path = example['all_messages']
    all_msgs = pickle.load(open(all_msgs_path, 'rb'))
    msgs = all_msgs['user_agent']
    return msgs

def construct_negative_example_in_mantis_format(example, id, metadata, add_task_inst=True):
    msgs = load_msgs_in_example(example)
    init_msg = msgs[0]
    images = eval(example['images']) if isinstance(example['images'], str) else example['images']
    kw = "image" if len(images) <= 1 else "images"
    prefix_templates = ["By examining the {kw}, ", 
                        "By analyzing the {kw} closely, ", 
                        "By looking at the {kw}, ", 
                        "By observing the {kw}, ", 
                        "By studying the {kw}, "]
    answer_templates = ["I can conclude that the answer is {answer}.", 
                        "the answer is {answer}.", 
                        "the answer seems to be {answer}.", 
                        "I think the answer is {answer}.", 
                        "I believe the answer is {answer}."]
    prefix_template = np.random.choice(prefix_templates)
    answer_template = np.random.choice(answer_templates)
    
    thought = prefix_template.format(kw=kw) + answer_template.format(answer=example['answer'])
    terminate_call = {
        "thought": thought,
        "actions": [{"name": "Terminate", "arguments": {"answer": example["answer"]}}],
    }
    final_msg = {'role': 'assistant', 'content': [{'type': 'text', 'text': json.dumps(terminate_call)}]}
    new_msgs = [init_msg, final_msg]
    # construct a new example with only the user input msg and the final answer msg
    example['all_messages'] = new_msgs
    # keep only the input images
    example['images'] = [image for image in images if not image.startswith("/export/share/zixianma/data/mma_execute/results")]
    return convert_openai_data_to_mantis_format(example, id, metadata, add_task_inst=add_task_inst)
    

def convert_iv_data_to_mantis_format(example, id, metadata, add_task_inst=True, add_obs_prefix=True, add_obs_suffix=True):
    mantis_example = {'id': f"{metadata['dataset']}-{id}", 'conversation': [], 'metadata': metadata, 'images': []}
    convos = []
    obs_prefix = "OBSERVATION:\n"
    obs_suffix = "\nThe OBSERVATION can be incomplete or incorrect, so please be critical and decide how to make use of it. If you've gathered sufficient information to answer the question, call Terminate with the final answer. Now, please generate the response for the next step."
    for i, msg in enumerate(example['conversations']):
        if i == 0 and msg['from'] == "human" and add_task_inst:
            convo_content = metadata['task_instruction'] + '\n'
        else:
            convo_content = "" 
        convo_content += msg['value']
        
        if i > 0 and msg['from'] == 'human': # process observation string
            if add_obs_prefix:
                convo_content = obs_prefix + convo_content
            if add_obs_suffix:
                convo_content = convo_content + obs_suffix
        new_msg = {'role': 'assistant' if msg['from'] == 'gpt' else 'user', 'content': convo_content}
        convos.append(new_msg)
    mantis_example['conversation'] = convos
    
    new_images_root_path = "/export/agentstudio-family/zixian/InstructVerse/data/new_images"
    image_paths = []
    for image in example['image']:
        if image.lower().find("vg") != -1:
            if image.lower().startswith("vg/"): # already in vg/
                image_path = os.path.join("/export/share/ayan/data", image)
            else:
                image_path = os.path.join("/export/share/ayan/data/vg", image)
        else:
            data_subset = metadata['data_subset'] if 'data_subset' in metadata else metadata['dataset'].split("_")[0]
            image_path = os.path.join(new_images_root_path, data_subset, image) # dataset name is v2-tool_mc but the image path has v2-tool only
        if not os.path.exists(image_path):
            print(f"Image path not found: {image_path}")
            return None
        image_paths.append(image_path)
    mantis_example['images'] = image_paths
    return mantis_example
    

def convert_openai_data_to_mantis_format(example, id, metadata, add_task_inst=True):
    mantis_example = {'id': f"{metadata['dataset']}-{id}", 'conversation': [], 'metadata': metadata, 'images': []}
    
    images = eval(example['images']) if isinstance(example['images'], str) else example['images']
    msgs = load_msgs_in_example(example)
    convos = []
    total_image_num = 0
    for i, msg in enumerate(msgs):
        convo = {}
        convo['role'] = msg['role']
        if add_task_inst and i == 0 and convo['role'] == "user":
            convo_content = metadata['task_instruction'] + '\n'
        else:
            convo_content = "" 
        
        for j, content in enumerate(msg['content']):
            if content['type'] == "text":

                text = content['text']
                convo_content += text + '\n'
            elif content['type'] == "image_url":
                pil_image = content['image_url']['url']
                image_path = images[total_image_num]
                if  metadata['dataset'].startswith('mantis-coinstruct') or metadata['dataset'] in ['mantis-spot-the-diff', 'mantis-birds-to-words', 'mantis-dreamsim', 'mantis-iconqa', 'mantis-multi_vqa', 'mantis-nlvr2', 'mantis-lrv_multi']:
                    image_path = image_path.replace("/export/share/jieyu/mantis_data", "/export/agentstudio-family/zixian/mantis_data") # metadata['dataset'].startswith('mantis-llava_665k_multi')
                    images[total_image_num] = image_path
                image = Image.open(image_path).convert("RGB")
                arr1 = np.array(pil_image)
                arr2 = np.array(image)
                assert np.array_equal(arr1, arr2), f"image1: {arr1}\nimage2: {arr2}"
                convo_content += f'<image>\n' #image-{image_id}:
                total_image_num += 1
            else:
                raise NotImplementedError
        
        if convo['role'] == 'assistant':
            convo_content = reformat_text(convo_content)
            try:
                parsed_content = json.loads(convo_content)
            except Exception as e:
                print(f"{id} CONVO:", convo_content)
                print(e)
                return None
        convo['content'] = convo_content
        convos.append(convo)
    if total_image_num != len(images):
        print(f"found different image nums: {total_image_num} != {len(images)}")
        return None
    mantis_example['conversation'] = convos
    mantis_example['images'] = images
    return mantis_example

def process_mantis_example(example, ex_id, turn_id, metadata, add_task_inst=True):
    mantis_example = {'id': f"{metadata['dataset']}-{ex_id}-{turn_id}", 'conversation': [], 'metadata': metadata, 'images': []}
    images = [elem['path'] for elem in example['images']]
    convo = example['conversation']
    convo_pos = turn_id * 2
    question = convo[convo_pos]['content'].replace("<image>", "").strip()
    answer = convo[convo_pos+1]['content']
    
    subset_name = metadata['dataset'].replace("mantis-", "")
    
    if metadata['dataset'].startswith('mantis-coinstruct') or metadata['dataset'] in ['mantis-spot-the-diff', 'mantis-birds-to-words', 'mantis-dreamsim', 'mantis-iconqa', 'mantis-multi_vqa', 'mantis-nlvr2', 'mantis-lrv_multi']:
        if metadata['dataset'].startswith('mantis-coinstruct'):
            subset_name = metadata['dataset'].split("-")[1] 
        image_dir = os.path.join("/export/agentstudio-family/zixian/mantis_data", subset_name)
    else:
        if metadata['dataset'].startswith("mantis-llava_665k_multi-"):
            subset_name = metadata['dataset'].split("-")[1] # llava_665k_multi
        image_dir = os.path.join("/export/share/jieyu/mantis_data", subset_name)
    # if not os.path.exists(image_dir):
    #     image_dir = os.path.join("/export/einstein-vision-hs/manlis_xgenmm/Mantis-Instruct", subset_name)
    #     if not os.path.exists(image_dir):
    #         return None
    msgs = [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
    image_paths = []
    convos = []
    for msg in msgs:
        role = msg['role']
        convo = msg.copy()
        convo['content'] = msg['content']
        if role == 'user':
            img_prefix_str = ""
            for i, image in enumerate(images):
                img_prefix_str += f"<image> " # image-{i}:
                img_path = os.path.join(image_dir, image)
                if not os.path.exists(img_path):
                    print(f"{img_path} not found.")
                    return None
                image_paths.append(img_path)
            convo['content'] = img_prefix_str + "\n" + convo['content']
            if add_task_inst:
                convo['content'] =  metadata['task_instruction'] + '\n' + convo['content']
        convos.append(convo)
    mantis_example['conversation'] = convos
    mantis_example['images'] = image_paths
    return mantis_example

def convert_cauldron_to_mantis_format(example, ex_id, turn_id, metadata, add_task_inst=True):
    mantis_example = {'id': f"{metadata['dataset']}-{ex_id}-{turn_id}", 'conversation': [], 'metadata': metadata, 'images': []}
    images = eval(example['images']) if isinstance(example['images'], str) else example['images']
    image_paths = []
    convos = []
    msg = example['texts'][turn_id]
    img_base_path = os.path.join(INPUT_IMAGE_PATH, metadata['dataset'])
    for role in ['user', 'assistant']:
        convo = {}
        convo['role'] = role
        if role == 'user':
            convo['content'] = msg[role]
            img_prefix_str = ""
            for i, image in enumerate(images): # these are PIL images
                img_prefix_str += f"<image> "
                img_path = os.path.join(img_base_path, str(ex_id), f"image-{i}.jpg") # png
                # try:
                if not os.path.exists(img_path):
                    print(img_path)
                    image = image.convert("RGB")
                    image.save(img_path)
                # image_from_path = Image.open(img_path).convert("RGB")
                # print(image, image_from_path)
                # arr1 = np.array(image)
                # arr2 = np.array(image_from_path)
                # difference = np.abs(arr1 - arr2)
                # max_diff = difference.max(axis=(0, 1))
                # sum_difference = np.sum(difference)
                # assert np.array_equal(arr1, arr2), f"image1: {arr1.shape}\nimage2: {arr2.shape}\nsum diff: {sum_difference}\nmax: {max_diff}"
                # except Exception as err:
                #     print(f"Error: {err}")
                #     return None
                image_paths.append(img_path)
            
            convo['content'] = img_prefix_str + convo['content']
            if add_task_inst:
                convo['content'] =  metadata['task_instruction'] + '\n' + convo['content']
        else:
            convo['content'] = msg[role]
        convos.append(convo)
    mantis_example['conversation'] = convos
    mantis_example['images'] = image_paths
    return mantis_example

def save_data_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Saved {len(data)} to {filename}")
    

def count_non_terminate_tools(row):
    tools = [tool['name'] for tool in eval(row['called_tools'])]
    if "Terminate" in tools:
        tools.remove("Terminate")
    return len(tools)


def convert_mantis_example_into_llava_format(example):
    new_example = {'sample_id': example['id'], 'image': example['images'], 'metadata': example['metadata']}
    new_convos = []
    for convo in example['conversation']:
        new_convo = {'from': 'human' if convo['role'] == 'user' else 'gpt', 'value': convo['content']}
        new_convos.append(new_convo)
    new_example['conversations'] = new_convos
    new_example['choice_list'] = None
    return new_example