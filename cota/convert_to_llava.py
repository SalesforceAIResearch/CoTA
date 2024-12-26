import os 
import argparse 
from mma.data_utils import *
from mma.config import *
from tqdm import trange, tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', required=True, type=str, help="a string representing a unique model.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    train_data_base_path = os.path.dirname(args.data_file)
    ds_name = os.path.basename(args.data_file).replace(".json", "")
    if ds_name.find("mantis") > -1:
        ds_name = ds_name.replace("mantis", "llava")
    else:
        ds_name = ds_name + "_llava"
    
    mantis_data = json.load(open(args.data_file, 'r'))
    llava_data = []
    for example in tqdm(mantis_data):
        llava_example = convert_mantis_example_into_llava_format(example)
        llava_data.append(llava_example)
    output_file = os.path.join(train_data_base_path, f"{ds_name}.json")
    save_data_to_json(llava_data, output_file)

if __name__ == "__main__":
    main()