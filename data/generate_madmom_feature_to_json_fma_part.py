import sys
import os
import json
from pathlib import Path
import argparse

import numpy as np
from madmom.features.key import CNNKeyRecognitionProcessor, KEY_LABELS
from madmom.features.beats import RNNBeatProcessor
from madmom.features.tempo import TempoEstimationProcessor

def main(args):
    fma_root_dir = args.fma_root
    json_path = args.json_path

    proc_key = CNNKeyRecognitionProcessor()
    proc_beat = RNNBeatProcessor()
    proc_tempo = TempoEstimationProcessor(fps=100)

    if os.path.exists(json_path):
        with open(json_path, "r") as jsonFile:
            feature_dict = json.load(jsonFile)
    else:
        feature_dict = {}

    subfolders = os.listdir(fma_root_dir)
    exts = ['mp3', 'wav']
    for subfolder in subfolders:
        audio_dir = os.path.join(fma_root_dir, subfolder)
        if not os.path.isdir(audio_dir):
            continue
        if int(subfolder) < args.folder_id_start or int(subfolder) >= args.folder_id_end:
            print("Folder id start: ", args.folder_id_start)
            print("Folder id end: ", args.folder_id_end)
            print("This folder if: ", int(subfolder))
            print("Not in the range of parsing, skipped")
            continue

        raw_audio_paths = [p for ext in exts for p in Path(f'{audio_dir}').glob(f'*.{ext}')]
        write_json_name_buff = []
        write_json_every = 10
        for i_path, audio_path in enumerate(raw_audio_paths):
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]
            audio_name = audio_name.split("_")[0]
            try:
                if os.path.getsize(audio_path) / 1000 > 300:
                    print(f"Processing {audio_name}")
                    if audio_name not in feature_dict:
                        feature_dict[audio_name] = {}
                        print(f"Created {audio_name} for json")
                    else:
                        print(f"Found {audio_name} in the json given")

                    if "madmom_key" not in feature_dict[audio_name]:
                        key_likelihood = proc_key(audio_path.as_posix())
                        key_decision = np.argmax(key_likelihood)
                        key_decision_str = KEY_LABELS[key_decision]
                        print(key_decision_str)
                        feature_dict[audio_name]["madmom_key"] = int(key_decision)
                    else:
                        print("madmom key feature already exists:", feature_dict[audio_name]["madmom_key"])

                    if "madmom_tempo" not in feature_dict[audio_name]:
                        beat_decision = proc_beat(audio_path.as_posix())
                        print(beat_decision)
                        tempo_decision = proc_tempo(beat_decision)[0][0]
                        print(tempo_decision)
                        feature_dict[audio_name]["madmom_tempo"] = int(tempo_decision)
                    else:
                        print("madmom key feature already exists:", feature_dict[audio_name]["madmom_tempo"])

                write_json_name_buff.append(audio_name)
                if i_path % write_json_every == write_json_every - 1 or i_path == len(raw_audio_paths) - 1:
                    with open(json_path, "w") as jsonFile:
                        json.dump(feature_dict, jsonFile)

                    print(f"Wrote {write_json_name_buff} to json")
                    write_json_name_buff = []
            except Exception as e:
                print(e)
                print(f"file {audio_path} is corrupted...  Loading another one instead")
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating h5 dac clap dataset.')
    parser.add_argument(
        '-fma-root', type=str,
        help='the folder saving mp3 or wav files'
    )
    parser.add_argument(
        '--json-path', type=str, nargs='?',
        help='the path that feature is saved'
    )
    parser.add_argument(
        '--folder-id-start', type=int,
    )
    parser.add_argument(
        '--folder-id-end', type=int,
    )

    args = parser.parse_args()
    main(args)



