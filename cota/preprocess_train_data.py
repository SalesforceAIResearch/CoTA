import os
import json
import argparse
from tqdm import trange, tqdm
import pandas as pd
from datasets import load_dataset
from mma.data_utils import *
from mma.config import *

simple_task_goal = """[BEGIN OF GOAL] You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. You can use these actions: OCR, LocalizeObjects, GetObjects, EstimateRegionDepth, EstimateObjectDepth, Crop, ZoomIn, QueryLanguageModel, GetImageToImagesSimilarity, GetImageToTextsSimilarity, GetTextToImagesSimilarity, DetectFaces, QueryKnowledgeBase, Calculate, SolveMathEquation, Terminate. [END OF GOAL]"""

simple_direct_answer_prompt =  """You are a helpful assistant, and your goal is to answer the question based on the image(s)."""

    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="gpt-4-vision-preview", type=str, help="a string representing a unique model.")
    parser.add_argument('--dataset', default="RealWorldQA", type=str, help="a string representing a unique dataset to run the agent on.")
    parser.add_argument('--exp-id', default="0725-two-depths-no-image-sim-json-max-reply-20-seed-42", type=str, help="a string representing a unique exp.")
    parser.add_argument('--data-id', default=None, type=str, help="a string representing a unique data version, only for baseline data.")
    parser.add_argument('--original-format', default="openai", type=str, help="a string representing the original data format.")
    parser.add_argument('--save-llava-format', action='store_true', help="whether to save the data as llava format.")
    parser.add_argument('--split', default="both", type=str, choices=['tool', 'no_tool', 'both'], help="a string representing the data splits to include.")
    parser.add_argument('--subset', default="both", type=str, choices=['positive', 'negative', 'both'], help="a string representing the data subset to include.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    ds_name = args.dataset.lower() # "mmvp"

    failed_ds2ids = {}
    if args.original_format == "baseline":
        cache_filename = os.path.join(TRAIN_DATA_PATH, f"baseline_cache.json")
        if os.path.exists(cache_filename):
            baseline_cache = json.load(open(cache_filename, 'r'))
        else:
            baseline_cache = {}
        print(len(baseline_cache))
        filename = os.path.join(TRAIN_DATA_PATH, f"baseline_mantis_{args.data_id}.json")
        if os.path.exists(filename):
            mantis_data = json.load(open(filename, 'r'))
        else:
            mantis_data = []
            # prev_datasets = set([ex['metadata']['dataset'] for ex in mantis_data])
            # print(len(mantis_data), prev_datasets)
            rel_ids = json.load(open(f"/export/agentstudio-family/zixian/notebooks/rel_ids_{args.data_id}.json", "r"))
            mantis_data = []
            total_num = 0
            for ds, ex2turn_ids in rel_ids.items():
                print(ds)
                # if ds in prev_datasets:
                #     continue
                metadata = {'dataset': ds, 'task_instruction': simple_direct_answer_prompt}
                if ds.startswith("mantis"):
                    subset_name = ds.replace("mantis-", "")
                    if subset_name.startswith("coinstruct") or subset_name.startswith("llava_665k_multi"):
                        subset_name = subset_name.split("-")[0]
                    dataset = load_dataset("TIGER-Lab/Mantis-Instruct", subset_name, split="train", cache_dir=os.environ['HF_HUB_CACHE'])
                else:
                    dataset = load_dataset("HuggingFaceM4/the_cauldron", ds, split="train", cache_dir=os.environ['HF_HUB_CACHE'])
                ex_num = 0
                for ex_id, turn_ids in tqdm(ex2turn_ids.items()):
                    ex_id = int(ex_id)
                    example = dict(dataset[ex_id])
                    for turn_id in turn_ids:
                        # total_num += 1
                        # ex_num += 1
                        # print("EXAMPLE:", example)
                        full_cache_id = "-".join([ds, str(ex_id), str(turn_id)])
                        if full_cache_id in baseline_cache:
                            mantis_example = baseline_cache[full_cache_id]
                        else:
                            print(f"not found in cache:", full_cache_id)
                            if ds.startswith("mantis"):
                                mantis_example = process_mantis_example(example, ex_id, turn_id, metadata)
                            else:
                                mantis_example = convert_cauldron_to_mantis_format(example, ex_id, turn_id, metadata)
                        if mantis_example:
                            mantis_data.append(mantis_example)
                            if full_cache_id not in baseline_cache:
                                baseline_cache[full_cache_id] = mantis_example
                print(len(mantis_data))
                    # else:
                    #     # print(f"failed to process:", ds, ex_id, turn_id)
                    #     failed_ds2ids[ds] = failed_ds2ids[ds] + [(ex_id, turn_id)] if ds in failed_ds2ids else [(ex_id, turn_id)]
            #     print(f"Total number of examples in {ds}: {ex_num}")
            # print(f"Total number of examples: {total_num}")
            # print(failed_ds2ids)
            save_data_to_json(mantis_data, filename)
            save_data_to_json(baseline_cache, cache_filename)
        
        if args.save_llava_format:
            # mantis_data = json.load(open(os.path.join(base_path, f"mma_mantis_{data_id}.json")))
            llava_data = []
            for example in mantis_data:
                llava_example = convert_mantis_example_into_llava_format(example)
                llava_data.append(llava_example)
            output_file = os.path.join(TRAIN_DATA_PATH, f"baseline_llava_{args.data_id}.json")
            save_data_to_json(llava_data, output_file)
        
    elif args.original_format == "iv":
        metadata = {'dataset': ds_name, 'task_instruction': simple_task_goal}
        train_data_base_path = os.path.join(TRAIN_DATA_PATH, "0915-program")
        os.makedirs(train_data_base_path, exist_ok=True)
        
        base_path = "/export/agentstudio-family/zixian/InstructVerse/data/"
        iv_data = json.load(open(os.path.join(base_path, f"{ds_name}.json")))
        mantis_data = []
        for i in trange(len(iv_data)):
            example = iv_data[i]
            if 'modes' in example:
                metadata.update(example['metadata'])
            mantis_example = convert_iv_data_to_mantis_format(example, example['id'], metadata)
            if mantis_example:
                mantis_data.append(mantis_example)
        
        filename = os.path.join(train_data_base_path, f"mantis_{ds_name}.json")
        save_data_to_json(mantis_data, filename)
        
        if args.save_llava_format:
            llava_data = []
            for example in mantis_data:
                llava_example = convert_mantis_example_into_llava_format(example)
                llava_data.append(llava_example)
            output_file = os.path.join(train_data_base_path, f"llava_{ds_name}.json")
            save_data_to_json(llava_data, output_file)
        
    elif args.original_format == "openai":
        metadata = {'dataset': ds_name, 'task_instruction': simple_task_goal}
        train_data_base_path = os.path.join(TRAIN_DATA_PATH, args.exp_id)
        os.makedirs(train_data_base_path, exist_ok=True)
        train_data_base_path = os.path.join(train_data_base_path, ds_name)
        os.makedirs(train_data_base_path, exist_ok=True)
    
        exp_id = args.exp_id
        res_dir = f"/export/agentstudio-family/zixian/mma/prediction/{args.model}/{ds_name}/"
        data_path = os.path.join(res_dir, f"{exp_id}.jsonl")

        os.makedirs(train_data_base_path, exist_ok=True)
        result_image_folder = os.path.join(RESULT_PATH, args.model, ds_name, exp_id)
        input_image_folder = os.path.join(INPUT_IMAGE_PATH, ds_name)
        
        jsonl_file_path = os.path.join(res_dir, f"{exp_id}_openai_result.jsonl")
        excel_file_path = os.path.join(res_dir, f"{exp_id}_gpt-4-turbo.xlsx")
        
        if os.path.exists(jsonl_file_path):
            data = pd.read_json(jsonl_file_path, lines=True)
        elif os.path.exists(excel_file_path):
            data = pd.read_excel(excel_file_path)
        else:
            print(f"File not found at: {jsonl_file_path} or {excel_file_path}")
            return 
            # raise NotImplementedError

        cols = list(data.columns)
        total_num = len(data)
        print(f"Total number: {total_num}")
        
        data['num_tools'] = data.apply(lambda row: count_non_terminate_tools(row), axis=1)
        tool_data = data[data['num_tools'] > 0]
        no_tool_data = data[data['num_tools'] == 0]
        all_data = {'tool': tool_data, 'no_tool': no_tool_data}
        all_stats = {'total_num': total_num, 'tool_rate': round(len(tool_data) / total_num, 3)}
        for k, data in all_data.items():
            correct_data = data
            sub_total_num = len(data)
            print(f"{k} total: {sub_total_num}")
            
            if 'score' in cols:
                correct_data = data[(data['score'] == True) | (data['score'] >= 0.5)]
            elif 'hit' in cols: # MMBench, MMMU, MMStar
                correct_data = data[data['hit'] == 1.0]
            
            if args.subset in ["both", "positive"]:
                correct_num = len(correct_data)
                print(f"{k} correct: {correct_num}")
                mantis_data = []
                for i in trange(len(correct_data)):
                    row = correct_data.iloc[i, :].to_dict()
                    id_str = int(row['index']) if not isinstance(row['index'], str) else row['index']
                    try:
                        mantis_example = convert_openai_data_to_mantis_format(row, id_str, metadata)
                    except FileNotFoundError as e:
                        # this might occur if an image is not found
                        print(f"File not found")
                        continue
                    if mantis_example:
                        mantis_data.append(mantis_example)

                final_num = len(mantis_data)
                print(f"{k} final correct: {final_num}")
            
                stats = {
                    "acc": round(correct_num / sub_total_num, 3) if sub_total_num > 0 else 0, 
                    "parse_rate": round(final_num / correct_num, 3) if correct_num > 0 else 0, 
                    "overall_rate": round(final_num / sub_total_num, 3) if sub_total_num > 0 else 0
                }
                all_stats[k] = stats
                filename = os.path.join(train_data_base_path, f"{ds_name}_{k}.json")
                save_data_to_json(mantis_data, filename)

                for k, v in all_stats.items():
                    print(f"{k}: {v}")
                save_data_to_json(all_stats, os.path.join(train_data_base_path, f"{ds_name}_stats.json"))

            if args.subset in ["both", "negative"]:
                # data = pd.read_json(data_path, lines=True)
                incorrect_data = data[~data.index.isin(correct_data.index)]
                print(f"{k} incorrect: {len(incorrect_data)}")
                mantis_neg_data = []
                for i in trange(len(incorrect_data)):
                    row = incorrect_data.iloc[i, :].to_dict()
                    id_str = int(row['index']) if not isinstance(row['index'], str) else row['index']
                    try:
                        mantis_neg_example = construct_negative_example_in_mantis_format(row, id_str, metadata)
                    except FileNotFoundError as e:
                        # this might occur if an image is not found
                        print(f"File not found")
                        continue
                    if mantis_neg_example:
                        mantis_neg_data.append(mantis_neg_example)
                final_neg_num = len(mantis_neg_data)
                print(f"{k} final neg: {final_neg_num}")
                neg_filename = os.path.join(train_data_base_path, f"{ds_name}_{k}_neg.json")
                save_data_to_json(mantis_neg_data, neg_filename)
            
    
if __name__ == "__main__":
    main()
    # all_datasets = ['mmvp', 'realworldqa', 'mmvet', 'cv-bench'] #, 'blink'
    # base_path = "/export/agentstudio-family/zixian/data"
    # all_data = []
    # for ds in all_datasets:
    #     data_path = os.path.join(base_path, f"{ds}.json")
    #     data = json.load(open(data_path, "r"))
    #     all_data += data
    # print(len(all_data))
    # save_data_to_json(all_data, os.path.join(base_path, "mma_data_all.json"))