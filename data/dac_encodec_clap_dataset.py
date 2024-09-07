import sys
sys.path.append("..")

from pathlib import Path
import os
import random
import numpy as np
import math

import h5py
import json

from datetime import datetime

import torch
from torch.utils.data import Dataset
import torchaudio

from encodec.utils import convert_audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import librosa
import audio_metadata
from tinytag import TinyTag

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

SPECTROGRAM_NORMALIZATION_FACTOR = 100

def get_true_audio_name(audioname):
    # if "[" in audioname and "]" in audioname: # for musiccaps
    #     return audioname
    # else:
    #     return audioname.split("_")[0]
    return audioname.split("_")[0]

'''
    Read audio files and convert them into dac tokens
'''
class DacEncodecClapDataset(Dataset):
    def __init__(
        self,
        audio_folder,
        dac_model,
        encodec_model = None,
        clap_model = None,
        exts = ['mp3', 'wav'],
        min_sec = 10,
        start_silence_sec = 0,
        skip_every_sec = None, # 15, if watermarks exist, have one chunk between each 2 watermarks
        skip_sec = 1.596-0.259,
        skip_start_sec = 15.259,
        chunk_dur_sec = 60,
        mel_freq_bins = 128,
        mel_window_size = 4096,
        mel_hop_size = 2048,
        min_size = 300,  # in KB
        percent_start_end = (None, None),
        no_audio_chunk = False,
        only_chroma = False,
    ):
        super().__init__()
        self.audio_folder = audio_folder
        self.dac_model = dac_model
        self.encodec_model = encodec_model
        self.clap_model = clap_model

        if encodec_model is not None:
            self.encodec_device = next(encodec_model.parameters()).device
            self.sample_rate_encodec = encodec_model.sample_rate
        if dac_model is not None:
            self.sample_rate_dac = dac_model.sample_rate
        if clap_model is not None:
            self.clap_device = next(clap_model.parameters()).device
            self.sample_rate_clap = 48000

        self.no_audio_chunk = no_audio_chunk

        self.sample_rate = 48000
        self.min_sec = min_sec

        self.mel_freq_bins = mel_freq_bins
        self.mel_window_size = mel_window_size
        self.mel_hop_size = mel_hop_size
        
        self.start_silence_sec = start_silence_sec
        self.skip_every_sec = skip_every_sec
        self.skip_sec = skip_sec
        self.skip_start_sec = skip_start_sec
        self.chunk_dur_sec = chunk_dur_sec
        if skip_every_sec is not None:
            assert skip_every_sec - skip_sec >= chunk_dur_sec, "cannot get chunks that do not contain watermark"
        
        self.raw_audio_paths = [p for ext in exts for p in Path(f'{audio_folder}').glob(f'*.{ext}')]
        self.raw_audio_paths_temp = []

        if percent_start_end[0] is None or percent_start_end[1] is None:
            i_raw_start = 0
            i_raw_end = len(self.raw_audio_paths)
        elif percent_start_end[0] >= 0 and percent_start_end[1] <= 100:
            assert percent_start_end[0] < percent_start_end[1]
            i_raw_start = int(percent_start_end[0] / 100 * len(self.raw_audio_paths))
            i_raw_end = int(percent_start_end[1] / 100 * len(self.raw_audio_paths))
        else:
            i_raw_start = 0
            i_raw_end = len(self.raw_audio_paths)

        for i_raw, audio_path in enumerate(self.raw_audio_paths):
            audio_filename = os.path.basename(audio_path)
            audio_name = os.path.splitext(audio_filename)[0]
            audio_name = get_true_audio_name(audio_name)
            if i_raw >= i_raw_start and i_raw < i_raw_end:
                if os.path.getsize(audio_path)/1000 > min_size:
                    self.raw_audio_paths_temp.append(audio_path)
                else:
                    print(audio_name, "is too short")
            # print(i_raw, audio_name, "is not in the percentage range, skipped")

        self.raw_audio_paths = self.raw_audio_paths_temp[:]
        self.audio_paths = self.raw_audio_paths[:]

        self.use_dac = True
        self.use_encodec = True
        self.use_clap = True
        if self.encodec_model is None:
            self.use_encodec = False
        if self.dac_model is None:
            self.use_dac = False
        if self.clap_model is None:
            self.use_clap = False

        self.recompute_feature = False

        # parse audio duration and get chunks to load
        # self.get_chunks()

    def get_chunks(self):
        start_silence_sec = self.start_silence_sec
        skip_every_sec = self.skip_every_sec
        skip_sec = self.skip_sec
        skip_start_sec = self.skip_start_sec
        chunk_dur_sec = self.chunk_dur_sec
        self.chunk_starts_pieces = [] # list of list
        self.chunk_ends_pieces = [] # list of list
        self.raw_durs = []
        self.audio_paths_temp = []
        for audio_path in self.audio_paths:
            audio_filename = os.path.basename(audio_path)
            audio_ext = os.path.splitext(audio_filename)[1]
            try:
                if "wav" in audio_ext:
                    raw_dur = audio_metadata.load(audio_path).streaminfo['duration']
                elif "mp3" in audio_ext:
                    raw_dur = TinyTag.get(audio_path).duration
                else:
                    print(f"{audio_path} is not in supported audio type, discarded")
                    continue
            except Exception as e:
                print(e)
                print(f"An error encountered in chunking file {audio_path}, skipped")
                continue

            if raw_dur < self.min_sec:
                print(f"{audio_path} is shorter than {self.min_sec} sec, discarded")
                continue

            self.audio_paths_temp.append(audio_path)
            self.raw_durs.append(raw_dur)
            chunk_starts = [start_silence_sec]
            if skip_every_sec is not None:
                chunk_starts_to_add = [s for s in np.arange(skip_start_sec+skip_sec, raw_dur-chunk_dur_sec, skip_every_sec)]
            else:
                chunk_starts_to_add = [s for s in np.arange(start_silence_sec+chunk_dur_sec, raw_dur-chunk_dur_sec, chunk_dur_sec)]

            chunk_starts += chunk_starts_to_add
            chunk_ends = [s+chunk_dur_sec for s in chunk_starts]

            self.chunk_starts_pieces.append(chunk_starts[:])
            self.chunk_ends_pieces.append(chunk_ends[:])

        self.audio_paths = self.audio_paths_temp[:]

        # assert (durations_sec > 0).all(), "there is an audio shorter than {} sec".format(start_silence_sec)
        num_chunks_per_file = [len(chunks) for chunks in self.chunk_starts_pieces]
        print("num audio paths:", len(self.raw_audio_paths))
        print("num valid audio paths:", len(self.audio_paths))
        self.num_chunks_cumsum = np.cumsum(num_chunks_per_file).astype(int)
        self.num_chunks_cumsum = np.insert(self.num_chunks_cumsum, 0, 0)
        self.num_chunks = self.num_chunks_cumsum[-1]
    
    def __len__(self):
        return self.num_chunks

    def find_file_id(self, index):
        file_id_low = 0
        file_id_high = len(self.num_chunks_cumsum)
        if file_id_high == 1:
            return 0
        else:
            while file_id_low < file_id_high:
                file_id_mid = math.floor((file_id_low + file_id_high)/2)
                
                this_chunk_id = self.num_chunks_cumsum[file_id_mid]
                next_chunk_id = self.num_chunks_cumsum[file_id_mid+1]
                if this_chunk_id <= index and next_chunk_id > index:
                    return file_id_mid
                elif this_chunk_id > index:
                    file_id_high = file_id_mid
                elif next_chunk_id <= index:
                    file_id_low = file_id_mid
                else:
                    assert 0, "invalid cumsum array"

    def get_rvq_latent_from_audio(self, wav, sample_rate = None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.use_dac:
            wav_dac = convert_audio(
                wav, sample_rate, self.sample_rate_dac, 1
            ).unsqueeze(0).to(self.dac_model.device) # [1, 1, n_samples]
        if self.use_encodec:
            wav_encodec = convert_audio(
                wav, sample_rate, self.sample_rate_encodec, self.encodec_model.channels
            ).unsqueeze(0).to(self.encodec_device)

        with torch.no_grad():
            if self.use_dac:
                _, codes, latents, _, _ = self.dac_model.encode(wav_dac) # [1, n_codebooks = 9, n_frames], [1, 8x9, n_frames]
                print(codes[0].shape, latents[0].shape)
            else:
                codes, latents = [None], [None]
            if self.use_encodec:
                encodec_codes = self.encodec_model.encode(wav_encodec)[0]
                encodec_latents = self.encodec_model.model.quantizer.decode(encodec_codes).reshape(-1, encodec_codes.shape[-1])
                print(encodec_codes[0].shape, encodec_latents.shape)
            else:
                encodec_codes, encodec_latents = [None], None

        return codes[0], latents[0], encodec_codes[0], encodec_latents
        # [9, n_frames] (int), [72, n_frames] (float), [4, n_frames] (int), (4, 128, n_frames) (float)

    def get_wav_through_file_id_and_relative_chunk_id(self, file_id, relative_index):
        assert relative_index >= 0, "invalid find file id function"
        audio_path = self.audio_paths[file_id]
        start_sec = self.chunk_starts_pieces[file_id][relative_index]
        sample_rate = torchaudio.info(audio_path).sample_rate
        start_frame = int(start_sec * sample_rate)
        dur_frame = int(self.chunk_dur_sec * sample_rate)
        # wav = librosa.load(audio_path, sr=self.sample_rate, offset=start_sec, duration=self.chunk_dur_sec)[0]
        wav, sr = torchaudio.load(audio_path, frame_offset = start_frame, num_frames = dur_frame)
        return wav, sr

    def get_wav_through_chunk_id(self, index):
        file_id = self.find_file_id(index)
        index_start = self.num_chunks_cumsum[file_id]
        relative_index = index - index_start
        return self.get_wav_through_file_id_and_relative_chunk_id(file_id, relative_index)

    @torch.no_grad()
    def get_rvq_latents_clap_from_wav(self, wav, sample_rate = None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        wav_clap = convert_audio(wav, sample_rate, self.sample_rate_clap, 1)

        dac_rvq, dac_latents, encodec_rvq, encodec_latents = self.get_rvq_latent_from_audio(wav, sample_rate)
        if self.use_clap:
            clap = self.clap_model.get_audio_embedding_from_data(x=wav_clap, use_tensor=True)
        else:
            clap = None

        return dac_rvq, dac_latents, encodec_rvq, encodec_latents, clap

    @torch.no_grad()
    def __getitem__(self, index):
        wav, sample_rate = self.get_wav_through_chunk_id(index)
        return self.get_rvq_latents_clap_from_wav(wav, sample_rate)

    def get_text_data_dict(self, audio_file_id):
        pass

    def get_spotify_meta_data_dict(self, audio_name):
        pass

    def get_text_feat_meta_data_dict(self, audio_name):
        pass

    def save_audio_text_to_h5_single(self, file_id, target_dir, skip_existing = False):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print("Target dir", target_dir, "is created!")

        audio_path = self.audio_paths[file_id]
        filename = Path(audio_path).name
        basename = os.path.splitext(filename)[0]
        out_h5_filename = f"{basename}.hdf5"
        out_h5_path = os.path.join(target_dir, out_h5_filename)

        # if skip_existing and os.path.exists(out_h5_path):
        #     print(f"{out_h5_path} already exists. Skipped")
        #     return

        print("=====================")

        print(file_id, ": Parsing", basename)

        # save text feat for FMA
        if hasattr(self, 'text_feat_metadata') and self.clap_model is not None:
            print("------saving text feat meta data------")
            with h5py.File(out_h5_path, 'a') as f:
                audio_name = get_true_audio_name(basename)
                data_dict = self.get_text_feat_meta_data_dict(audio_name)

                if "metadata" not in f:
                    grp = f.create_group("metadata")
                else:
                    grp = f["metadata"]

                for attr in data_dict:
                    if attr not in grp:
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("archived", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                    else:
                        del grp[attr]
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("updated", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                print("\n")

        if self.no_audio_chunk:
            return

        chunk_starts = self.chunk_starts_pieces[file_id]
        for relative_index in range(len(chunk_starts)):
            chunkname = f'{relative_index}'

            print(file_id, ": Parsing", basename, "chunk num", relative_index, "from", audio_path)

            # save audio
            print("------saving audio data------")
            with h5py.File(out_h5_path, 'a') as f:
                if chunkname not in f or not skip_existing:
                    wav, sample_rate = self.get_wav_through_file_id_and_relative_chunk_id(file_id, relative_index)
                    wav_spec = convert_audio(
                        wav, sample_rate, self.sample_rate, 1
                    ).squeeze().numpy()  # [n_samples]

                    powergram = librosa.stft(
                        y=wav_spec, n_fft=self.mel_window_size,
                        hop_length=self.mel_hop_size, window='hann', center=True,
                        pad_mode='constant'
                    )
                    powergram = np.abs(powergram)**2
                    spectrogram = librosa.feature.melspectrogram(
                        S=powergram, sr=self.sample_rate, n_mels=self.mel_freq_bins
                    )
                    chromagram = librosa.feature.chroma_stft(S=powergram, sr=self.sample_rate)
                    spectrogram_dB = librosa.power_to_db(spectrogram)
                    spectrogram_dB = spectrogram_dB / SPECTROGRAM_NORMALIZATION_FACTOR

                    if (chunkname not in f) or self.recompute_feature:
                        dac_rvq, dac_latents, encodec_rvq, encodec_latents, audio_clap = self.get_rvq_latents_clap_from_wav(
                            wav, sample_rate
                        )
                    else:
                        dac_rvq = torch.tensor(np.array(f[chunkname]['dac_rvq'])) if 'dac_rvq' in f[chunkname] else None
                        dac_latents = torch.tensor(np.array(f[chunkname]['dac_latents'])) if 'dac_latents' in f[chunkname] else None
                        encodec_rvq = torch.tensor(np.array(f[chunkname]['encodec_rvq'])) if 'encodec_rvq' in f[chunkname] else None
                        encodec_latents =torch.tensor( np.array(f[chunkname]['encodec_latents'])) if 'encodec_latents' in f[chunkname] else None
                        audio_clap = torch.tensor(int16_to_float32(np.array(f[chunkname]['audio_clap']))).unsqueeze(0) if 'audio_clap' in f[chunkname] else None

                    print(f"Got chunk number {relative_index}")
                    data_dict = {}
                    if dac_rvq is not None:
                        data_dict['dac_rvq'] = dac_rvq.cpu().numpy().astype(int)
                        data_dict['dac_frame_len'] = dac_rvq.shape[-1]
                        print("Got dac RVQ code with shape", data_dict['dac_rvq'].shape)
                    if dac_latents is not None:
                        data_dict['dac_latents'] = dac_latents.cpu().numpy().astype(np.float32)
                        print("Got dac latent with shape", data_dict['dac_latents'].shape)
                    if encodec_rvq is not None:
                        data_dict['encodec_rvq'] = encodec_rvq.cpu().numpy().astype(int)
                        data_dict['encodec_frame_len'] = encodec_rvq.shape[-1]
                        print("Got encodec RVQ code with shape", data_dict['encodec_rvq'].shape)
                    if encodec_latents is not None:
                        data_dict['encodec_latents'] = encodec_latents.cpu().numpy().astype(np.float32)
                        print("Got encodec latent with shape", data_dict['encodec_latents'].shape)
                    if audio_clap is not None:
                        data_dict['audio_clap'] = float32_to_int16(audio_clap.cpu().numpy())[0, :]
                        print("Got audio clap with shape", data_dict['audio_clap'].shape)

                    data_dict['spectrogram'] = float32_to_int16(spectrogram_dB)
                    print("Got mel spectrogram with shape", spectrogram_dB.shape)
                    data_dict['chroma'] = float32_to_int16(chromagram)
                    print("Got chromagram with shape", chromagram.shape)

                    if chunkname not in f:
                        grp = f.create_group(chunkname)
                    else:
                        grp = f[chunkname]

                    for attr in data_dict:
                        if attr not in grp:
                            grp.create_dataset(attr, data=data_dict[attr])
                            print("archived", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                        else:
                            del grp[attr]
                            grp.create_dataset(attr, data=data_dict[attr])
                            print("updated", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                    print("\n")
                else:
                    print("chunk already exists or audio chunk disabled")
                    print("\n")


    def save_audio_text_to_h5_multiple(self, target_dir, skip_existing = False, file_id_sel = None):
        if file_id_sel is None:
            file_id_sel = range(len(self.audio_paths))
        for file_id in file_id_sel:
            try:
                self.save_audio_text_to_h5_single(file_id, target_dir, skip_existing=skip_existing)
            except Exception as ex:
                trace = []
                tb = ex.__traceback__
                while tb is not None:
                    trace.append({
                        "filename": tb.tb_frame.f_code.co_filename,
                        "name": tb.tb_frame.f_code.co_name,
                        "lineno": tb.tb_lineno
                    })
                    tb = tb.tb_next
                print(str({
                    'type': type(ex).__name__,
                    'message': str(ex),
                    'trace': trace
                }))

class DacEncodecClapTextFeatDataset(DacEncodecClapDataset):
    def __init__(self, *args, json_path = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.json_path = json_path
        if json_path is not None:
            with open(json_path) as f:
                self.text_feat_metadata = json.load(f)

            self.audio_paths = []
            self.audio_names_from_json = list(self.text_feat_metadata.keys())
            for audio_path in self.raw_audio_paths:
                audio_filename = os.path.basename(audio_path)
                audio_name = os.path.splitext(audio_filename)[0]
                audio_name = get_true_audio_name(audio_name)
                if audio_name in self.audio_names_from_json:
                    self.audio_paths.append(audio_path)
                else:
                    print(audio_name, "is not in the json metadata")

        # parse audio duration and get chunks to load
        if not self.no_audio_chunk:
            self.get_chunks()

    def get_text_feat_meta_data_dict(self, audio_name):
        metadata_dict = self.text_feat_metadata[audio_name]

        print(f"Got metadata of audio {audio_name}")

        data_dict = {
            'madmom_key': np.array(metadata_dict['madmom_key']),
            'madmom_tempo': np.array(metadata_dict['madmom_tempo']),
        }
        if 'text' in metadata_dict:
            print("Found text description")
            data_dict['text'] = metadata_dict['text'].encode("ISO-8859-1", "ignore")
            print("Got text", metadata_dict['text'])
            text_clap = self.clap_model.get_text_embedding([metadata_dict['text'], ""])
            print("Got CLAP text emb with dummy shape", text_clap.shape)
            data_dict['text_clap'] = float32_to_int16(text_clap[0])

        if 'salmonn_text' in metadata_dict:
            print("Found salmonn_text description")
            data_dict['salmonn_text'] = metadata_dict['salmonn_text'].encode("ISO-8859-1", "ignore")
            print("Got salmonn_text", metadata_dict['salmonn_text'])
            salmonn_text_clap = self.clap_model.get_text_embedding([metadata_dict['salmonn_text'], ""])
            print("Got CLAP text emb with dummy shape", salmonn_text_clap.shape)
            data_dict['salmonn_text_clap'] = float32_to_int16(salmonn_text_clap[0])

        if 'chatgpt_texts' in metadata_dict:
            print("Found chatgpt_texts description")
            data_dict['chatgpt_texts'] = np.array([caption.encode("ISO-8859-1", "ignore") for caption in metadata_dict['chatgpt_texts']])
            print("Got chatgpt_texts", metadata_dict['chatgpt_texts'])
            chatgpt_texts_clap = self.clap_model.get_text_embedding(metadata_dict['chatgpt_texts'] + [""])
            print("Got CLAP text emb with dummy shape", chatgpt_texts_clap.shape)
            data_dict['chatgpt_text_clap'] = float32_to_int16(chatgpt_texts_clap[:-1])

        return data_dict

'''
    Read the h5 files in a folder
    getting the dac_latents&rvq, encodec_latents&rvq, audio_clap, and meta data / text if applicable
    Can be set to random mode, where the dataset_size should be specified
'''
class DacEncodecClapDatasetH5(Dataset):
    def __init__(
        self,
        h5_dir,
        dac_frame_len,
        encodec_frame_len,
        chroma_frame_len=None,
        dataset_size=1000,
        random_load=True,
        random_text=True,
        inpaint=False,
    ):
        super().__init__()
        self.random_load = random_load

        self.h5_dir = h5_dir
        self.dac_frame_len = dac_frame_len
        self.encodec_frame_len = encodec_frame_len
        self.chroma_frame_len = chroma_frame_len
        if self.chroma_frame_len % 8 != 0:
            self.chroma_frame_len = self.chroma_frame_len - (self.chroma_frame_len % 8)

        print("Reading dac h5 file metadata...")
        self.h5_filenames = [filename for filename in os.listdir(h5_dir) if ".hdf5" in filename]
        self.basenames = [os.path.splitext(filename)[0] for filename in self.h5_filenames]

        print("Num files:", len(self.h5_filenames))
        self.random_text = random_text

        if not self.random_load:
            self.num_chunks_per_file = []
            for file_id in range(len(self.h5_filenames)):
                file_name = self.h5_filenames[file_id]
                file_path = os.path.join(self.h5_dir, file_name)
                with h5py.File(file_path, "r") as f:
                    num_chunks = len(f)
                    if "text" in f:
                        num_chunks = num_chunks - 1
                    if "metadata" in f:
                        num_chunks = num_chunks - 1
                    self.num_chunks_per_file.append(num_chunks)

            self.num_chunks_per_file = np.array(self.num_chunks_per_file)
            self.num_chunks_per_file_cumsum = np.cumsum(self.num_chunks_per_file)
            print("Num chunks:", self.num_chunks_per_file_cumsum[-1])
        else:
            self.dataset_size = dataset_size  # as loading is random, the size is dummy

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # self.text_emb_dim = 1024
        # self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        self.inpaint = inpaint
        self.total_runtime = 0
        self.total_visits = 0

    def __len__(self):
        if not self.random_load:
            return self.num_chunks_per_file_cumsum[-1]
        else:
            return self.dataset_size

    def get_file_id_from_chunk_id(self, index):
        return np.argmax(self.num_chunks_per_file_cumsum > index)

    @torch.no_grad()
    def get_objects(self, index):
        if self.random_load:
            file_id = random.randint(0, len(self.h5_filenames) - 1)
        else:
            file_id = self.get_file_id_from_chunk_id(index)

        file_name = self.h5_filenames[file_id]
        file_path = os.path.join(self.h5_dir, file_name)

        with h5py.File(file_path, "r") as f:
            num_chunks = len(f)
            if "text" in f:
                num_chunks = num_chunks - 1
            if "metadata" in f:
                num_chunks = num_chunks - 1

            if num_chunks <= 0:
                return self.get_objects(index + 1)

            return_dict = {}

            if "metadata" in f:
                metadata_h5 = f["metadata"]
                if "madmom_key" in metadata_h5:
                    return_dict["madmom_key"] = np.array(metadata_h5['madmom_key'])
                if "madmom_tempo" in metadata_h5:
                    return_dict["madmom_tempo"] = np.array(metadata_h5['madmom_tempo'])

                if "text" in metadata_h5:
                    return_dict["text"] = metadata_h5["text"][()].decode("ISO-8859-1")
                else:
                    return_dict["text"] = ""

                # print("Raw caption in dataset: ", return_dict["text"])

                if "text_clap" in metadata_h5:
                    return_dict["text_clap"] = int16_to_float32(np.array(metadata_h5["text_clap"]))
                # else:
                #     return_dict["text_clap"] = int16_to_float32(np.array(f["0"]['audio_clap']))

                if "chatgpt_texts" in metadata_h5:
                    return_dict["chatgpt_texts"] = np.array(
                        [str_code.decode("ISO-8859-1") for str_code in np.array(metadata_h5["chatgpt_texts"])]
                    )
                else:
                    return_dict["chatgpt_texts"] = np.array([])

                # print("Chatgpt caption in dataset: ", return_dict["chatgpt_texts"])

                # DEBUG
                # if "chatgpt_text_clap" in metadata_h5:
                #     return_dict["chatgpt_text_clap"] = int16_to_float32(np.array(metadata_h5["chatgpt_text_clap"]))
                # else:
                #     return_dict["chatgpt_text_clap"] = int16_to_float32(np.array(f["0"]['audio_clap']))

            if self.random_load:
                chunk_id = str(random.randint(0, num_chunks - 1))
                # chunk_id = str(0) # debug for overfit test
            else:
                if file_id == 0:
                    relative_chunk_id = int(index)
                else:
                    relative_chunk_id = int(index - self.num_chunks_per_file_cumsum[file_id - 1])
                chunk_id = str(relative_chunk_id)

            chunk_name = self.basenames[file_id] + f"_chunk_{chunk_id}"
            return_dict["name"] = chunk_name

            start_proportion = None
            if 'dac_frame_len' in f[chunk_id]:
                dac_frame_len_file = np.array(f[chunk_id]['dac_frame_len'])
                if dac_frame_len_file < self.dac_frame_len:
                    print("dac_frame_len_file, self.dac_frame_len:", dac_frame_len_file, self.dac_frame_len)
                    if index < self.__len__():
                        return self.get_objects(index + 1)
                    else:
                        return self.get_objects(index - 1)
                    print(f"Chunk {chunk_name} is too short, loading another")
                dac_frame_start = random.randint(0, dac_frame_len_file - self.dac_frame_len)
                start_proportion = dac_frame_start / dac_frame_len_file
                length_proportion = self.dac_frame_len /dac_frame_len_file
                # frame_start = 0 # debug for overfit test
                dac_latents = f[chunk_id]['dac_latents'][:, dac_frame_start:dac_frame_start + self.dac_frame_len]
                dac_rvq = f[chunk_id]['dac_rvq'][:, dac_frame_start:dac_frame_start + self.dac_frame_len]
                return_dict["dac_rvq"] = dac_rvq
                return_dict["dac_latents"] = dac_latents

            if 'encodec_frame_len' in f[chunk_id]:
                encodec_frame_len_file = np.array(f[chunk_id]['encodec_frame_len'])
                if encodec_frame_len_file < self.encodec_frame_len:
                    if index < self.__len__():
                        return self.get_objects(index + 1)
                    else:
                        return self.get_objects(index - 1)
                    print(f"Chunk {chunk_name} is too short, loading another")
                encodec_frame_start = random.randint(0, encodec_frame_len_file - self.encodec_frame_len)
                start_proportion = encodec_frame_start / encodec_frame_len_file
                encodec_latents = f[chunk_id]['encodec_latents'][:,encodec_frame_start:encodec_frame_start + self.encodec_frame_len]
                encodec_rvq = f[chunk_id]['encodec_rvq'][:, encodec_frame_start:encodec_frame_start + self.encodec_frame_len]
                return_dict["encodec_rvq"] = encodec_rvq
                return_dict["encodec_latents"] = encodec_latents

            assert start_proportion is not None

            if 'audio_clap' in f[chunk_id]:
                audio_clap = int16_to_float32(np.array(f[chunk_id]['audio_clap']))
                return_dict["audio_clap"] = audio_clap

            if 'clap' in f[chunk_id]:
                clap = int16_to_float32(np.array(f[chunk_id]['clap']))
                return_dict["clap"] = clap

            if "spectrogram" in f[chunk_id]:
                raw_spectrogram = int16_to_float32(np.array(f[chunk_id]['spectrogram']))
                spec_raw_len = raw_spectrogram.shape[-1]
                spec_start = int(spec_raw_len * start_proportion)

                return_dict["spectrogram"] = raw_spectrogram[..., spec_start:spec_start+self.chroma_frame_len]

            if "chroma" in f[chunk_id]:
                raw_chroma = int16_to_float32(np.array(f[chunk_id]['chroma']))
                chroma_raw_len = raw_chroma.shape[-1]
                chroma_start = int(chroma_raw_len * start_proportion)

                return_dict["chroma"] = raw_chroma[..., chroma_start:chroma_start+self.chroma_frame_len]

            # if "text" in f:
            #     text_clap = int16_to_float32(np.array(f["text"]['clap']))
            #     text = [str_bytes.decode("ISO-8859-1") for str_bytes in np.array(f["text"]['texts'])]
            #
            #     if self.random_text:
            #         text_id = random.randint(0, text_clap.shape[0] - 1)
            #     return_dict["text_clap"] = text_clap[text_id]
            #     return_dict["text"] = text[text_id]

            if self.inpaint:
                dac_mask = np.ones_like(dac_latents)
                mask_start = self.dac_frame_len // 4
                mask_end = (3*self.dac_frame_len) // 4
                dac_mask[:, mask_start:mask_end] *= 0
                return_dict["dac_inpaint_mask"] = dac_mask.astype(bool)

                encodec_mask = np.ones_like(dac_latents)
                mask_start = self.encodec_frame_len // 4
                mask_end = (3*self.encodec_frame_len) // 4
                encodec_mask[:, mask_start:mask_end] *= 0
                return_dict["encodec_inpaint_mask"] = encodec_mask.astype(bool)

        return_dict = self.encode_random_text_to_t5(return_dict)
        # return_dict["randomized_text"] = "" # DEBUG
        # print("Randomized text in return dict: ", return_dict["randomized_text"])
        # print(type(return_dict["randomized_text"]))
        return_dict.pop("chatgpt_texts")


        # with torch.autocast(device_type="cpu", dtype=torch.float32):

        enc = self.tokenizer(
            return_dict["randomized_text"],
            return_tensors="pt", truncation=True, padding='longest'
        )# .to(self.t5_model.device)
        # print("t5_input_ids", enc["input_ids"].shape)
        # print("t5_attention_mask", enc["input_ids"].shape)
        return_dict["t5_input_ids"] = enc["input_ids"].squeeze(0).cpu().numpy()
        return_dict["t5_attention_mask"] = enc["attention_mask"].squeeze(0).cpu().numpy()

        # # print("In dataloader tokenized:", enc)
        # # print("In dataloader T5 model device:", self.t5_model.device)
        # text_emb_seq = self.t5_model.encoder(
        #     input_ids=enc["input_ids"],
        #     attention_mask=enc["attention_mask"],
        #     return_dict=True,
        # ).last_hidden_state
        # # print(emb.shape)
        # # print("In dataloader T5 emb mean, shape:", text_emb_seq.mean(), text_emb_seq.shape)
        #
        # return_dict["t5_emb"] = text_emb_seq.permute(0, 2, 1).squeeze(0).cpu().numpy() # [D, L]

        return return_dict

    def encode_random_text_to_t5(self, data_dict):
        possible_texts = []
        if 'text' in data_dict:
            if len(data_dict['text']) > 0:
                possible_texts.append(data_dict['text'])
        if 'chatgpt_texts' in data_dict:
            if len(data_dict['chatgpt_texts']) > 0:
                possible_texts += data_dict['chatgpt_texts'].tolist()
        if len(possible_texts) == 0:
            data_dict["randomized_text"] = f"Description missing"
        else:
            num_string = len(possible_texts)
            selected_sentence = possible_texts[np.random.randint(num_string)]

            data_dict["randomized_text"] = str(selected_sentence) + f""

        # data_dict["randomized_text"] = [data_dict["randomized_text"]] # np.array(data_dict["randomized_text"]).tobytes()

        return data_dict

    def __getitem__(self, index):
        start = datetime.now()
        return_dict = self.get_objects(index)
        dur = datetime.now() - start
        self.total_visits += 1
        self.total_runtime += dur.total_seconds()
        return return_dict
