import os
import argparse
import pandas as pd
import json

def main(args):
    data_frame = pd.read_excel(args.xlsx_path, index_col=None)
    file_ids = data_frame["ID"]
    file_starts = data_frame["start_s"]
    file_ends = data_frame["end_s"]
    texts = data_frame["Description"]
    is_evals = data_frame["is_audioset_eval"]

    num_files = len(data_frame)

    out_dict = {}
    for i_data in range(num_files):
        # if is_evals[i_data]:
        print(file_ids[i_data], file_starts[i_data], file_ends[i_data])
        file_name = "["+str(file_ids[i_data])+"]-["+str(int(file_starts[i_data]))+"-"+str(int(file_ends[i_data]))+"]"
        out_dict[file_name] = {"text": texts[i_data]}
        print(file_name, texts[i_data])
        if i_data % 100 == 0 or i_data == num_files-1:
            with open(args.out_json_path, "w") as jsonFile:
                json.dump(out_dict, jsonFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC RVQ diffusion.')
    parser.add_argument(
        '-xlsx-path', type=str,
        help='the path storing the excel'
    )
    parser.add_argument(
        '-out-json-path', type=str,
        help='the path outputing the json with text caption'
    )
    args = parser.parse_args()
    main(args)
