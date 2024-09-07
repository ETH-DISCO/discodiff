import sys
sys.path.append("..")

from pathlib import Path
import os
import random
import numpy as np

import json
import librosa
from tinytag import TinyTag

from torch.utils.data import Dataset


'''
    Read audio files and convert them into dac tokens
'''
class AudioDatasetDISCO200K(Dataset):
    def __init__(
        self,
        audio_folder,
        json_path,  # for spotify metadata
        exts=['mp3', 'wav'],
        chunk_sec=25,
        random_chunk=True,
        preload = True,
        load_rand = 1.0,
    ):
        super().__init__()
        self.audio_folder = audio_folder
        self.random_chunk = random_chunk
        self.chunk_sec = chunk_sec

        self.raw_audio_paths = [p for ext in exts for p in Path(f'{audio_folder}').glob(f'*.{ext}')]
        self.audio_paths = []

        self.json_path = json_path
        with open(json_path) as f:
            self.spotify_metadata = json.load(f)

        # parse audio according to json
        self.audio_id_to_metadata_id = []
        self.audio_names_from_json = list(self.spotify_metadata.keys())
        for audio_path in self.raw_audio_paths:
            if load_rand is not None:
                prob = random.random()
                use_this = prob < load_rand
            else:
                use_this = True

            if use_this:
                audio_filename = os.path.basename(audio_path)
                audio_name = os.path.splitext(audio_filename)[0]
                audio_name = audio_name.split("_")[0]
                if audio_name in self.audio_names_from_json and os.path.getsize(audio_path)/1000 > 300:
                    metadata_id = self.audio_names_from_json.index(audio_name)
                    self.audio_id_to_metadata_id.append(metadata_id)
                    self.audio_paths.append(audio_path)
                    print(f"loaded {audio_name}")
                else:
                    print(audio_name, "is not in the json metadata or is too short")

        self.preload = preload
        if preload:
            print("Preload activated.")
            self.waveforms = []
            self.audio_names = []
            self.spotify_metadata_for_waveforms = []
            for i_audio in range(len(self.audio_paths)):
                try:
                    audio_path = self.audio_paths[i_audio]
                    # metadata_id = self.audio_id_to_metadata_id[i_audio]
                    # audio_name = self.audio_names_from_json[metadata_id]
                    filename = Path(audio_path).name
                    basename = os.path.splitext(filename)[0]
                    audio_name = basename.split("_")[0]
                    metadata_dict = self.spotify_metadata[audio_name]

                    waveform, sample_rate = librosa.load(audio_path)
                    num_chunks = int(len(waveform) / sample_rate / chunk_sec)
                    for i_chunk in range(num_chunks):
                        sample_start = i_chunk * sample_rate
                        sample_end = (i_chunk + 1) * sample_rate
                        self.waveforms.append(waveform[sample_start:sample_end])
                        self.audio_names.append(audio_name)
                        self.spotify_metadata_for_waveforms.append(metadata_dict)

                except Exception as e:
                    print(e)
                    print(f"file {audio_path} is corrupted...  Loading another one instead")
                    i_audio += 1
                    continue
        else:
            print("Preload not activated. Dataset loading may take more time.")


    def __len__(self):
        if not self.preload:
            return len(self.audio_paths)
        else:
            return len(self.waveforms)

    def __getitem__(self, audio_id):
        if not self.preload:
            i = 0
            time_tries = 10
            while i < time_tries:
                try:
                    audio_path = self.audio_paths[audio_id]
                    # metadata_id = self.audio_id_to_metadata_id[audio_id]
                    # audio_name = self.audio_names_from_json[metadata_id]
                    audio_filename = os.path.basename(audio_path)
                    audio_name = os.path.splitext(audio_filename)[0]
                    audio_name = audio_name.split("_")[0]
                    metadata_dict = self.spotify_metadata[audio_name]

                    # if self.random_chunk:
                    #     raw_dur = TinyTag.get(audio_path).duration
                    #     start_sec = random.randint(0, int(raw_dur - self.chunk_sec))
                    # else:
                    #     start_sec = 0

                    start_sec = 0
                    waveform, sample_rate = librosa.load(audio_path, offset=start_sec, duration=self.chunk_sec)

                except Exception as e:
                    print(e)
                    print(f"file {audio_name} is corrupted...  Loading another one instead")
                    i += 1
                    audio_id += 1
                    continue
        else:
            metadata_dict = self.spotify_metadata_for_waveforms[audio_id]
            waveform = self.waveforms[audio_id]
            audio_name = self.audio_names[audio_id]

        data_dict = {
            'waveform': waveform,
            'name': audio_name,
            'acousticness': np.array(metadata_dict['acousticness']),
            'danceability': np.array(metadata_dict['danceability']),
            'energy': np.array(metadata_dict['energy']),
            'instrumentalness': np.array(metadata_dict['instrumentalness']),
            'key': np.array(metadata_dict['key']),
            'liveness': np.array(metadata_dict['liveness']),
            'loudness': np.array(metadata_dict['loudness']),
            'mode': np.array(metadata_dict['mode']),
            'speechiness': np.array(metadata_dict['speechiness']),
            'tempo': np.array(metadata_dict['tempo']),
            'time_signature': np.array(metadata_dict['time_signature']),
            'valence': np.array(metadata_dict['valence']),
        }

        return data_dict