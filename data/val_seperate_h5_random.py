import argparse
import os
from pathlib import Path
import random

def main(args):
    h5_dir = args.h5_dir
    val_size = args.val_size

    val_dir = os.path.join(h5_dir, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    exts = ["h5", "hdf5"]
    raw_h5_paths = [p for ext in exts for p in Path(f'{h5_dir}').glob(f'*.{ext}')]
    sampled_h5_paths = random.sample(raw_h5_paths, val_size)
    for h5_path in sampled_h5_paths:
        os.system("mv {} {}".format(h5_path, val_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in seperating a val h5 dataset from the whole.')
    parser.add_argument(
        '-h5-dir', type=str,
        help='the folder saving h5 files'
    )
    parser.add_argument(
        '--val-size', type=int, default=400,
        help='the num of files in the validation set'
    )
    args = parser.parse_args()
    main(args)