import os
import argparse
import h5py

def main(args):
    hdf5_files = os.listdir(args.hdf5_dir)
    for file_name in hdf5_files:
        file_path = os.path.join(args.hdf5_dir, file_name)
        if not (".hdf5" in file_path or ".h5" in file_path):
            continue
        with h5py.File(file_path, "r") as f:
            num_chunks = len(f)
            if "text" in f:
                num_chunks = num_chunks - 1
            if "metadata" in f:
                num_chunks = num_chunks - 1

            if num_chunks <= 0:
                continue

            file_del_flag = False
            for chunk_id_num in range(num_chunks):
                chunk_id = str(chunk_id_num)

                if "chroma" not in f[chunk_id]:
                    file_del_flag = True
                    print(f"File {file_name} has one chunk that does not contain chroma, deleted")
                    break

            if file_del_flag:
                os.remove(file_path)
            else:
                print(f"File {file_name} has chroma, retained")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '--hdf5-dir', type=str, default='',
        help='the directory that h5 dataset files are saved'
    )
    args = parser.parse_args()
    main(args)
