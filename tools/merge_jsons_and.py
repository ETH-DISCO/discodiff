import os
import json
from pathlib import Path
import argparse

def main(args):
    json_path1 = args.json_path1
    json_path2 = args.json_path2
    out_json_path = args.out_json_path


    with open(json_path1, "r") as jsonFile:
        dict1 = json.load(jsonFile)
    with open(json_path2, "r") as jsonFile:
        dict2 = json.load(jsonFile)

    dict2_keys = list(dict2.keys())
    dict2_base_strs = [attr.split("_")[0] for attr in dict2_keys]

    for attr in dict1:
        base_str = attr.split("_")[0]
        if base_str in dict2_base_strs:
            index_dict2 = dict2_base_strs.index(base_str)
            attr_dict2 = dict2_keys[index_dict2]
            component_dict2 = dict2[attr_dict2]

            for subattr in component_dict2:
                if subattr not in dict1[attr]:
                    dict1[attr][subattr] = component_dict2[subattr]
            print("Merged dict, getting", dict1[attr])
        else:
            print(f"{base_str} not found in {json_path2}")

    with open(out_json_path, "w") as jsonFile:
        json.dump(dict1, jsonFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Merge two jsons into one. Each json is a dict of dicts. 
        For example:
        dict1 = {
            "name1": {"text": "A simple song"},
        }
        dict2 = {
            "name1_chunk1": {"key": 0, "tempo": 120},
            "name1_chunk2": {"key": 0, "tempo": 120},
            "name2": {"key": 4, "tempo": 90},
        }
        out_dict = {
            "name1": {"key": 0, "tempo": 120, "text": "A simple song"},
        }
        '''
    )
    parser.add_argument(
        '-json-path1', type=str,
    )
    parser.add_argument(
        '-json-path2', type=str,
    )
    parser.add_argument(
        '-out-json-path', type=str, default="merged_dict.json",
        help='the path that merged dict is saved'
    )

    args = parser.parse_args()
    main(args)