import os
import json
import torch
import argparse
import torchaudio
from encodec.utils import convert_audio
import laion_clap

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get CLAP audio encoder
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device=device) # need pip install transformers==4.30.0; if later version is installed, downgrade it to 4.30.0
    clap_model.load_ckpt(
        "./music_audioset_epoch_15_esc_90.14.pt"
    ) # download the default pretrained checkpoint.
    print("CLAP model loaded")

    if args.text_json_path is not None:
        with open(args.text_json_path) as f:
            text_feat_metadata = json.load(f)
    elif args.text_dir is not None:
        text_files = os.listdir(args.text_dir)
        text_feat_metadata = {}
        for filename in text_files:
            if len(filename)>4 and filename[-4:] == ".txt":
                audio_name = os.path.splitext(filename)[0].split("_")[0]
                text_file = os.path.join(args.text_dir, filename)
                with open(text_file, "r") as file:
                    caption = file.readline().strip()
                text_feat_metadata[audio_name] = {'text': caption}
    else:
        assert 0, "Should provide caption json file or caption directory of txt files."

    cosine_sim_list = []
    audio_files = os.listdir(args.audio_dir)
    with torch.no_grad():
        for audio_file in audio_files:
            audio_name = os.path.splitext(audio_file)[0].split("_")[0]
            audio_path = os.path.join(args.audio_dir, audio_file)
            if audio_name in text_feat_metadata:
                wav, sr = torchaudio.load(audio_path)
                wav_clap = convert_audio(wav, sr, 48000, 1)
                audio_clap = clap_model.get_audio_embedding_from_data(x=wav_clap, use_tensor=True).squeeze().cpu()
                text_clap = clap_model.get_text_embedding([text_feat_metadata[audio_name]['text'], ""])[0]
                text_clap = torch.tensor(text_clap).squeeze().cpu()

                cosine_similarity = torch.cosine_similarity(
                    text_clap.unsqueeze(0),
                    audio_clap.unsqueeze(0),
                    dim=1,
                    eps=1e-12,
                )
                print(audio_name, ":", cosine_similarity.item())

                cosine_sim_list.append(cosine_similarity.item())

    print("---final clap score---")
    print(sum(cosine_sim_list) / len(cosine_sim_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '-audio-dir', type=str, default='',
        help='the directory that model results are saved'
    )
    parser.add_argument(
        '-text-dir', type=str, nargs="?",
        help='the directory that captions are saved'
    )
    parser.add_argument(
        '-text-json-path', type=str, nargs="?",
        help='the json file that captions are saved'
    )
    args = parser.parse_args()
    main(args)