import os
import json
from pathlib import Path
import argparse

def main(args):
    json_dir = args.json_dir
    out_json_path = args.out_json_path

    json_paths = [p for p in Path(f'{json_dir}').glob(f'*.json')]

    out_dict = {}
    for json_path in json_paths:
        with open(json_path, "r") as jsonFile:
            this_dict = json.load(jsonFile)

        out_base_strs = [attr.split("_")[0] for attr in out_dict]
        this_base_strs = [attr.split("_")[0] for attr in this_dict]
        this_keys = list(this_dict.keys())

        for base_str in this_base_strs:
            if base_str not in out_base_strs:
                index_in_this_dict = this_base_strs.index(base_str)
                this_attr = this_keys[index_in_this_dict]
                out_dict[base_str] = this_dict[this_attr]
            else:
                print(f"{base_str} already in the previous json parsed")

    with open(out_json_path, "w") as jsonFile:
        json.dump(out_dict, jsonFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Merge multiple jsons into one. Each json is a dict of dicts. 
        For example:
        dict1 = {
            "name1": {"text": "A simple song"},
        }
        dict2 = {
            "name2": {"text": "Another simple song"},
        }
        out_dict = {
            "name1": {"text": "A simple song"},
            "name2": {"text": "Another simple song"},
        }
        '''
    )
    parser.add_argument(
        '-json-dir', type=str,
        help='the dir where all jsons are stored'
    )
    parser.add_argument(
        '-out-json-path', type=str, default="merged_dict.json",
        help='the path that merged dict is saved'
    )

    args = parser.parse_args()
    main(args)