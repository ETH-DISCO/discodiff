import argparse
from pathlib import Path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in DAC sanity check.')
    parser.add_argument(
        '-audio-dir', type=str,
    )
    parser.add_argument(
        '-save-path', type=str,
    )
    parser.add_argument(
        '-audio-name', type=str,
    )
    args = parser.parse_args()

    exts = ['mp3', 'wav']
    correct_audio_path = None
    for p in Path(f'{args.audio_dir}').glob('*.*'):
        if args.audio_name in p:
            correct_audio_path = p

    if correct_audio_path is not None:
        # _, filename = os.path.split(correct_audio_path)
        # target_audio_path = os.path.join(args.save_path, filename)
        cmd = f'cp {correct_audio_path} {args.save_path}'
        os.system(cmd)
    else:
        print(f"Input audio {args.audio_name} not found in the dataset rooted at {args.audio_dir}")
