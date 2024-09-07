import os
import json
import torch
import argparse
from pathlib import Path
from model import SALMONN
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--whisper_path", type=str, default=None)
    parser.add_argument("--beats_path", type=str, default=None)
    parser.add_argument("--vicuna_path", type=str, default=None)
    parser.add_argument(
        '-dataset-root', type=str,
        help='the folder saving folders of mp3 or wav files'
    )
    parser.add_argument(
        '--out-json-path', type=str, nargs='?',
        help='the path that feature is saved'
    )
    parser.add_argument(
        '--folder-id-start', type=int, default = 0
    )
    parser.add_argument(
        '--folder-id-end', type=int, default = 100
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path,
        low_resource=False
    )
    model.to(device)
    model.eval()
    prompt = "Describe the music with up to three sentences. Do indicate the style, genre, tempo and mood if possible."
    
    print("=====================================")
    
    root_dir = args.dataset_root
    subfolders = os.listdir(root_dir)
    json_path = args.out_json_path
    if os.path.exists(json_path):
        with open(json_path, "r") as jsonFile:
            feature_dict = json.load(jsonFile)
    else:
        feature_dict = {}
    
    exts = ['mp3', 'wav']
    for subfolder in subfolders:
        audio_dir = os.path.join(root_dir, subfolder)
        if not os.path.isdir(audio_dir):
            continue
        if int(subfolder) < args.folder_id_start or int(subfolder) >= args.folder_id_end:
            print("Folder id start: ", args.folder_id_start)
            print("Folder id end: ", args.folder_id_end)
            print("This folder is: ", int(subfolder))
            print("Not in the range of parsing, skipped")
            continue

        raw_audio_paths = [p for ext in exts for p in Path(f'{audio_dir}').glob(f'*.{ext}')]
        write_json_name_buff = []
        write_json_every = 10
    
        for i_path, audio_path in enumerate(raw_audio_paths):
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]
            audio_name = audio_name.split("_")[0]
            if os.path.getsize(audio_path) / 1000 <= 300:
                print(f"{audio_name} is too small, skipped")
            
            if audio_name not in feature_dict:
                feature_dict[audio_name] = {}
                print(f"Created {audio_name} for json")
            else:
                print(f"Found {audio_name} in the json given")
                
            if "salmonn_text" not in feature_dict[audio_name]:
                # for environment with cuda>=117
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    description = model.generate(audio_path, prompt=prompt)[0]
                    print(description)
                feature_dict[audio_name]["salmonn_text"] = description
            else:
                print("text already exists:", feature_dict[audio_name]["salmonn_text"])
                
            write_json_name_buff.append(audio_name)
            if i_path % write_json_every == write_json_every - 1 or i_path == len(raw_audio_paths) - 1:
                with open(json_path, "w") as jsonFile:
                    json.dump(feature_dict, jsonFile)

                print(f"Wrote {write_json_name_buff} to json")
                write_json_name_buff = []
                
    print("Output:")

