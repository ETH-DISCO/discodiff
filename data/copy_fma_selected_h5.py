import os
import argparse
from pathlib import Path

def main(args):
    selection_path = args.selection_path
    orig_fma_dir = args.orig_fma_dir
    out_fma_dir = args.out_fma_dir

    exts = ['h5', 'hdf5']
    # raw_h5_files = [p for ext in exts for p in Path(f'{orig_fma_dir}').glob(f'*.{ext}')]
    raw_h5_files = os.listdir(orig_fma_dir)
    with open(selection_path, 'r') as txt_file:
        selected_fma_ids = [line.rstrip() for line in txt_file.readlines()]
    selected_fma_ids = [this_id.split('.')[0].zfill(6) for this_id in selected_fma_ids]
    print("selected fma ids:", selected_fma_ids)

    selected_h5_files = []
    for h5_file in raw_h5_files:
        fma_id = h5_file.split('.')[0]
        if fma_id in selected_fma_ids and (".h5" in h5_file or ".hdf5" in h5_file):
            selected_h5_files.append(h5_file)

    for h5_file in selected_h5_files:
        h5_path = os.path.join(orig_fma_dir, h5_file)
        command = f"cp {h5_path} {out_fma_dir}"
        print(command)
        os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating madmom features from audio.')
    parser.add_argument(
        '-selection-path', type=str,
        help='the txt file indicating the fma files to select'
    )
    parser.add_argument(
        '-orig-fma-dir', type=str,
        help='the dir that stores the full fma files'
    )
    parser.add_argument(
        '-out-fma-dir', type=str,
        help='the dir that selected fma to be copied to'
    )

    args = parser.parse_args()
    main(args)