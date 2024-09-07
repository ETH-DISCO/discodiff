import os
import argparse
from frechet_audio_distance import FrechetAudioDistance

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-gt-dir", type=str)
    parser.add_argument("-eval-dir", type=str)
    parser.add_argument("--model-type", type=str, default="vggish")
    parser.add_argument("--save-gt-emb-path", type=str, nargs="?")
    parser.add_argument("--save-eval-emb-path", type=str, nargs="?")
    args = parser.parse_args()

    if args.model_type == "vggish":
        frechet = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=True
        )
    elif args.model_type == "PANN":
        frechet = FrechetAudioDistance(
            model_name="pann",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=True
        )
    elif args.model_type == "CLAP":
        frechet = FrechetAudioDistance(
            model_name="clap",
            sample_rate=48000,
            submodel_name="630k-audioset",  # for CLAP only
            verbose=True,
            enable_fusion=False,            # for CLAP only
        )
    else:
        assert 0, "FAD model type should be chosen from [vggish, PANN, CLAP]"

    if args.save_gt_emb_path is None:
        background_embds_path = None
    else:
        background_embds_path = args.save_gt_emb_path
    if args.save_eval_emb_path is None:
        eval_embds_path = None
    else:
        eval_embds_path = args.save_eval_emb_path

    fad_score = frechet.score(
        args.gt_dir,
        args.eval_dir,
        background_embds_path=background_embds_path,
        eval_embds_path=eval_embds_path,
        dtype="float32"
    )
    print("ground-truth audio dir:", args.gt_dir)
    print("validation audio dir:", args.eval_dir)
    print(fad_score)