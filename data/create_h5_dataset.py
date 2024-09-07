import argparse
import torch
import os
from dac_encodec_clap_dataset import DacEncodecClapTextFeatDataset

def create_h5_dataset_given_audio_dir(
    audio_dir, json_path, target_dir, dac_model, encodec_model, clap_model,
    percent_start_end = (None, None), skip_existing = False,
    chunk_dur_sec = 27, min_sec = 28,
    no_audio_chunk = False
):
    if os.path.isdir(audio_dir):
        print("----------------------------------")
        print(f"Parsing audio folder {audio_dir}")
        subdataset = DacEncodecClapTextFeatDataset(
            audio_folder = audio_dir,
            dac_model = dac_model,
            encodec_model = encodec_model,
            clap_model = clap_model,
            json_path = json_path,
            exts = ['mp3', 'wav'],
            start_silence_sec = 0,
            chunk_dur_sec = chunk_dur_sec,
            min_sec = min_sec,
            percent_start_end = percent_start_end,
            no_audio_chunk = no_audio_chunk
        )
        print("Dataset created")
        subdataset.save_audio_text_to_h5_multiple(target_dir, skip_existing = skip_existing)

def get_dac_encodec_clap(use_dac = True, use_encodec = True, use_clap = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_dac:
        import dac
        dac_model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(dac_model_path)
        dac_model.to(device)
    else:
        dac_model = None

    if use_encodec:
        from audiocraft.models import CompressionModel
        encodec_model = CompressionModel.get_pretrained('facebook/encodec_32khz')
        encodec_model.to(device)
    else:
        encodec_model = None

    if use_clap:
        import laion_clap
        clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base', device=device) # need pip install transformers==4.30.0; if later version is installed, downgrade it to 4.30.0
        clap_model.load_ckpt(
            "./music_audioset_epoch_15_esc_90.14.pt"
        ) # download the default pretrained checkpoint.
    else:
        clap_model = None

    return dac_model, encodec_model, clap_model

def main(args):
    audio_dir = args.audio_dir
    target_dir = args.target_dir
    json_path = args.json_path

    dac_model, encodec_model, clap_model = get_dac_encodec_clap(
        use_dac = not args.no_dac,
        use_encodec = not args.no_encodec,
        use_clap = not args.no_clap
    )

    percent_start_end = (args.percent_start, args.percent_end)
    create_h5_dataset_given_audio_dir(
        audio_dir, json_path, target_dir, dac_model, encodec_model, clap_model,
        percent_start_end = percent_start_end, skip_existing = args.skip_existing,
        chunk_dur_sec = args.chunk_dur_sec, min_sec = args.min_sec,
        no_audio_chunk=args.no_audio_chunk
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating h5 dac clap dataset.')
    parser.add_argument(
        '-audio-dir', type=str,
        help='the folder saving mp3 or wav files'
    )
    parser.add_argument(
        '--target-dir', type=str, default='',
        help='the directory that h5 data is saved'
    )
    parser.add_argument(
        '--json-path', type=str, nargs='?',
        help='the path that text feat metadata is saved'
    )
    parser.add_argument(
        '--percent-start', type=float, nargs='?',
    )
    parser.add_argument(
        '--percent-end', type=float, nargs='?',
    )
    parser.add_argument(
        '--no-dac', type=bool, default=False,
    )
    parser.add_argument(
        '--no-encodec', type=bool, default=False,
    )
    parser.add_argument(
        '--no-clap', type=bool, default=False,
    )
    parser.add_argument(
        '--skip-existing', type=bool, default=False,
    )
    parser.add_argument(
        '--chunk-dur-sec', type=float, default=27.0,
    )
    parser.add_argument(
        '--min-sec', type=float, default=28.0,
    )
    parser.add_argument(
        '--no-audio-chunk', type=bool, default=False,
    )
    args = parser.parse_args()
    main(args)