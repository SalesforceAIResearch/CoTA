import os
import json
import argparse
import pandas as pd
import numpy as np
from cota.config import *
from cota.data_utils import convert_mantis_example_into_llava_format, save_data_to_json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-id', default=None, type=str, help="a string representing a unique data version.")
    parser.add_argument('--exp-id', default="0915-15-tools-json-max-reply-10-seed-42", type=str, help="a string representing a unique exp.")
    parser.add_argument('--save', action='store_true', help="whether to save all the data.")
    parser.add_argument('--all', action='store_true', help="whether to use all the datasets.")
    parser.add_argument('--save-llava-format', action='store_true', help="whether to save the data as llava format.")
    parser.add_argument('--split', default="both", type=str, choices=['tool', 'no_tool', 'both'], help="a string representing the data splits to include.")
    parser.add_argument('--subset', default="both", type=str, choices=['positive', 'negative', 'both'], help="a string representing the data subset to include.")
    parser.add_argument('--max-image-num', default=None, type=int, help="an integer representing the max number of images in the kept example.")
    args = parser.parse_args()
    return args

def load_data_with_optional_max_image_num(data_path, max_image_num=None):
    data = json.load(open(data_path, "r"))
    if max_image_num:
        kept_data = []
        for ex in data:
            if len(ex['images']) > max_image_num:
                continue 
            kept_data.append(ex)
        return kept_data
    else:
        return data

def merge_all_data():
    args = get_args()
    base_path = os.path.join(TRAIN_DATA_PATH, args.exp_id)
    all_data = []
    included_datasets = []
    overall_stats = {}
    all_datasets = ALL_DATASETS if args.all else GOOD_DATASETS
    for ds in all_datasets:
        data_path = os.path.join(base_path, ds)
        if not os.path.exists(data_path):
            print(f"no {ds} data\n")
            continue
        overall_stats[ds] = {}
        included_datasets.append(ds)
        old_len = len(all_data)
        if args.split in ["tool", "both"]:
            tool_data_path = os.path.join(data_path, f"{ds}_tool.json")
            if os.path.exists(tool_data_path):
                tool_data = load_data_with_optional_max_image_num(tool_data_path, args.max_image_num)
                all_data += tool_data
                overall_stats[ds]['tool'] = len(tool_data)
            
            if args.subset in ["negative", "both"]:
                tool_neg_data_path = os.path.join(data_path, f"{ds}_tool_neg.json")
                if os.path.exists(tool_neg_data_path):
                    tool_neg_data = load_data_with_optional_max_image_num(tool_neg_data_path, args.max_image_num)
                    all_data += tool_neg_data
                    overall_stats[ds]['tool_neg'] = len(tool_neg_data)
                
        if args.split in ["no_tool", "both"]:
            no_tool_data_path = os.path.join(data_path, f"{ds}_no_tool.json")
            if os.path.exists(no_tool_data_path):
                no_tool_data = load_data_with_optional_max_image_num(no_tool_data_path, args.max_image_num) 
                all_data += no_tool_data
                overall_stats[ds]['no_tool'] = len(no_tool_data)
        
            if args.subset in ["negative", "both"]:
                
                no_tool_neg_data_path = os.path.join(data_path, f"{ds}_no_tool_neg.json")
                if os.path.exists(no_tool_neg_data_path):
                    no_tool_neg_data = load_data_with_optional_max_image_num(no_tool_neg_data_path, args.max_image_num)
                    all_data += no_tool_neg_data
                    overall_stats[ds]['no_tool_neg'] = len(no_tool_neg_data)

        new_len = len(all_data)
        overall_stats[ds]['total'] = new_len-old_len
        print(f"{ds}:", new_len-old_len)
        # assert len(all_data) == np.sum([overall_stats[ds]['total'] if ds in overall_stats else 0 for ds in all_datasets])
    print(f"Included {len(included_datasets)} datasets: {included_datasets}")
    print(len(all_data))
    return all_data, overall_stats

def main():
    args = get_args()
    mantis_data, overall_stats = merge_all_data()
    if not args.data_id:
        data_id = f"{args.split}_{args.subset}_{round(len(mantis_data) / 1000)}k"
    if args.max_image_num:
        data_id = f"max_{args.max_image_num}_img_" + data_id
    if not args.all:
        data_id = "filtered_" + data_id

    if args.save:
        base_path = os.path.join(TRAIN_DATA_PATH, args.exp_id)
        save_data_to_json(mantis_data, os.path.join(base_path, f"mma_mantis_{data_id}.json"))
        df = pd.DataFrame.from_dict(overall_stats, orient='index')
        df = df[df.total != 0]
        df.loc['all'] = df.sum(axis=0)
        df.to_csv(os.path.join(base_path, f"mma_{data_id}_stats.csv"))
        print(df.head())
    
        if args.save_llava_format:
            # mantis_data = json.load(open(os.path.join(base_path, f"mma_mantis_{data_id}.json")))
            llava_data = []
            for example in mantis_data:
                llava_example = convert_mantis_example_into_llava_format(example)
                llava_data.append(llava_example)
            output_file = os.path.join(base_path, f"mma_llava_{data_id}.json")
            save_data_to_json(llava_data, output_file)
        
if __name__ == "__main__":
    main()
    