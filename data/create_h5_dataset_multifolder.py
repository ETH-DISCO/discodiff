import argparse
import os
from create_h5_dataset import create_h5_dataset_given_audio_dir, get_dac_encodec_clap

def main(args):
    root_dir = args.root_dir
    target_dir = args.target_dir
    json_path = args.json_path

    dac_model, encodec_model, clap_model = get_dac_encodec_clap(
        use_dac = not args.no_dac,
        use_encodec = not args.no_encodec,
        use_clap = not args.no_clap
    )

    subfolders = os.listdir(root_dir)
    for subfolder in subfolders:
        audio_dir = os.path.join(root_dir, subfolder)
        if not os.path.isdir(audio_dir):
            continue
        if not ((args.folder_id_start is None) and (args.folder_id_end is None)):
            if int(subfolder) < args.folder_id_start or int(subfolder) >= args.folder_id_end:
                print("Folder id start: ", args.folder_id_start)
                print("Folder id end: ", args.folder_id_end)
                print("This folder if: ", int(subfolder))
                print("Not in the range of parsing, skipped")
                continue

        create_h5_dataset_given_audio_dir(
            audio_dir, json_path, target_dir, dac_model, encodec_model, clap_model,
            percent_start_end = (None, None), skip_existing = args.skip_existing,
            no_audio_chunk = args.no_audio_chunk
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating h5 dac clap dataset.')
    parser.add_argument(
        '-root-dir', type=str,
        help='the folder that contain subfolders saving mp3 or wav files'
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
        '--folder-id-start', type=int, nargs='?',
    )
    parser.add_argument(
        '--folder-id-end', type=int, nargs='?',
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
        '--no-audio-chunk', type=bool, default=False,
    )
    args = parser.parse_args()
    main(args)