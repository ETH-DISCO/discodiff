import os
import argparse
import torch
import pytorch_lightning as pl

from utils import load_state_dict_partial_primary_secondary, load_state_dict_partial_chroma
from pl_module_callbacks import LatentDiffusionCondModule
from train_conditional import sel_config, config_adjustments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '-primary-ckpt-path', type=str,
    )
    parser.add_argument(
        '-secondary-ckpt-path', type=str,
    )
    parser.add_argument(
        '-chroma-ckpt-path', type=str, nargs="?",
    )
    parser.add_argument(
        '-out-ckpt-path', type=str
    )
    parser.add_argument(
        '--rvq-pattern', type=str, default='parallel',
        help='choose from "parallel", "flattened" and "VALL-E"; default: "parallel"'
    )
    parser.add_argument(
        '--model-size', type=str, default='large',
        help='choose from "large" and "small"; default: "large"'
    )
    parser.add_argument(
        '--scheduler', type=str, default='handcrafted',
        help='choose from "handcrafted", "diffusers"; default: "handcrafted"'
    )
    args = parser.parse_args()
    config = sel_config(model_size = args.model_size, rvq_pattern = args.rvq_pattern)
    config = config_adjustments(config, scheduler = args.scheduler)
    diffusion_pl_module = LatentDiffusionCondModule(config)
    load_state_dict_partial_primary_secondary(
        diffusion_pl_module.state_dict(),
        torch.load(args.primary_ckpt_path, map_location=torch.device('cpu'))['state_dict'],
        torch.load(args.secondary_ckpt_path, map_location=torch.device('cpu'))['state_dict'],
    )
    print("Loaded checkpoint", args.primary_ckpt_path, "and", args.secondary_ckpt_path)
    if args.chroma_ckpt_path is not None:
        load_state_dict_partial_chroma(
            diffusion_pl_module.state_dict(),
            torch.load(args.chroma_ckpt_path, map_location=torch.device('cpu'))['state_dict'],
        )
        print("Loaded checkpoint", args.chroma_ckpt_path)

    trainer = pl.Trainer(accelerator="cpu")
    trainer.strategy.connect(diffusion_pl_module)
    trainer.save_checkpoint(args.out_ckpt_path)

    print("Saved checkpoint", args.out_ckpt_path)