from pathlib import Path
import os

from torch.utils.data import Dataset

import json
import librosa
from .dac_encodec_clap_dataset import DacEncodecClapTextFeatDataset


'''
    Read audio files of FMA
    Dataset structure:
    - Root
        - 000
            - file1.mp3
            - file2.mp3
        - 001 
            - file1.mp3
            - file2.mp3
        ...
'''
class AudioDatasetFMA(Dataset):
    def __init__(
        self,
        root_folder,
        exts=['mp3', 'wav'],
        preload = True,
        min_size = 300, # in KB
        specific_folder = None
    ):
        super().__init__()
        self.root_folder = root_folder
        self.raw_audio_paths = []
        subdirs = os.listdir(self.root_folder)
        for subdir in subdirs:
            audio_folder = os.path.join(self.root_folder, subdir)
            parse_this = os.path.isdir(audio_folder)
            if specific_folder is not None:
                parse_this = parse_this * (str(specific_folder) == subdir)
            if parse_this:
                self.raw_audio_paths += [p for ext in exts for p in Path(f'{audio_folder}').glob(f'*.{ext}')]

        self.audio_paths = []
        self.audio_names = []

        for audio_path in self.raw_audio_paths:
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]
            audio_name = audio_name.split("_")[0]
            if os.path.getsize(audio_path)/1000 > min_size:
                self.audio_paths.append(audio_path)
                self.audio_names.append(audio_name)
                print(f"loaded {audio_name}")
            else:
                print(audio_name, "is too short")

        self.preload = preload
        if preload:
            self.waveforms = []
            audio_names_tmp = []
            for i_audio in range(len(self.audio_paths)):
                try:
                    audio_path = self.audio_paths[i_audio]
                    waveform, sample_rate = librosa.load(audio_path)
                    self.waveforms.append(waveform)
                    audio_names_tmp.append(self.audio_names[i_audio])

                except Exception as e:
                    print(e)
                    print(f"file {audio_path} is corrupted...  Loading another one instead")
                    i_audio += 1
                    continue

            self.audio_names = audio_names_tmp[:]

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
                    waveform, sample_rate = librosa.load(audio_path)
                except Exception as e:
                    print(e)
                    print(f"file {audio_path} is corrupted...  Loading another one instead")
                    i += 1
                    audio_id += 1
                    continue
        else:
            waveform = self.waveforms[audio_id]

        return_dict = {
            "waveform": waveform,
            "name": self.audio_names[audio_id]
        }
        return return_dict
