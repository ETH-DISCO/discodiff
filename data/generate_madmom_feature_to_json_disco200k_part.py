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
    audio_dir = args.audio_dir
    json_path = args.json_path

    proc_key = CNNKeyRecognitionProcessor()
    proc_beat = RNNBeatProcessor()
    proc_tempo = TempoEstimationProcessor(fps=100)

    if os.path.exists(json_path):
        with open(json_path, "r") as jsonFile:
            feature_dict = json.load(jsonFile)
    else:
        feature_dict = {}

    exts = ['mp3', 'wav']
    raw_audio_paths = [p for ext in exts for p in Path(f'{audio_dir}').glob(f'*.{ext}')]
    write_json_name_buff = []
    write_json_every = 10

    num_files = len(raw_audio_paths)
    id_start = int(args.percent_start * num_files / 100)
    id_end = int(args.percent_end * num_files / 100)

    for i_path, audio_path in enumerate(raw_audio_paths):
        if id_start > i_path or id_end <= i_path:
            print(f"{audio_path} skipped")
            continue

        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        audio_name = audio_name.split("_")[0]
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

    with open(json_path, "w") as jsonFile:
        json.dump(feature_dict, jsonFile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating madmom features from audio.')
    parser.add_argument(
        '-audio-dir', type=str,
        help='the folder saving mp3 or wav files'
    )
    parser.add_argument(
        '--json-path', type=str, nargs='?',
        help='the path that feature is saved'
    )
    parser.add_argument(
        '--percent-start', type=float,
    )
    parser.add_argument(
        '--percent-end', type=float,
    )

    args = parser.parse_args()
    main(args)



