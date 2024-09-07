import sys
import os
import argparse

import torch

import pytorch_lightning as pl
import wandb

from configs.config_dac import (
    config_dac_parallel_large,
    config_dac_flattened_large,
    config_dac_valle_large,
    config_dac_parallel_small,
    config_dac_flattened_small,
    config_dac_valle_small,
)

from pl_module_callbacks import LatentDiffusionCondModule, DacCLAPDataModule
from pl_module_callbacks import ExceptionCallback, DemoCallback

def sel_config(model_size, rvq_pattern):
    if model_size == "large":
        if rvq_pattern == "parallel":
            config = config_dac_parallel_large
        elif rvq_pattern == "flattened":
            config = config_dac_flattened_large
        elif rvq_pattern == "VALL-E":
            config = config_dac_valle_large
        else:
            print("The RVQ pattern is not allowed")
            exit()
    elif model_size == "small":
        if rvq_pattern == "parallel":
            config = config_dac_parallel_small
        elif rvq_pattern == "flattened":
            config = config_dac_flattened_small
        elif rvq_pattern == "VALL-E":
            config = config_dac_valle_small
        else:
            print("The RVQ pattern is not allowed")
            exit()
    else:
        print("The model size string is not allowed")
        exit()

    return config

def set_save_path(save_path):
    save_path_ret = None if save_path == "" else save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Created result path", save_path)
    return save_path_ret

def set_device_accelerator():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu' # DEBUG
    print('Using device:', device)
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    return device, accelerator

def config_adjustments(config, **kwargs):
    for key, value in kwargs.items():
        if hasattr(config, key) and (value is not None):
            config.__dict__[key] = value
    return config

def main(args):
    # determine config type according to pattern
    config = sel_config(args.model_size, args.rvq_pattern)
    print(config)

    save_path = set_save_path(args.save_path)
    device, accelerator = set_device_accelerator()
    # torch.manual_seed(0)

    config = config_adjustments(
        config,
        batch_size = args.batch_size,
        prediction_type = args.prediction_type,
        scheduler = args.scheduler,
        frame_len_dac = args.frame_len_dac,
        max_epochs = args.max_epochs
    )

    config.chroma_frame_len = int(config.frame_len_dac / 3.687)
    config.chroma_frame_len = config.chroma_frame_len - (config.chroma_frame_len % 8)

    # get dataset
    data_module = DacCLAPDataModule(
        h5_dir = args.h5_dir,
        config = config,
        custum_dataset_size = args.dataset_size,
        val_h5_dir = args.val_h5_dir,
        another_h5_dir = args.another_h5_dir,
    )
    print("Dataset created")

    # get pl diffusion module (model defined inside)
    diffusion_pl_module = LatentDiffusionCondModule(
        config,
        ckpt_path = args.load_ckpt_path,
        only_train_primary = args.only_train_primary,
        only_train_secondary = args.only_train_secondary,
        only_train_chroma = args.only_train_chroma,
        no_clap_feat = args.no_clap_feat
    )
    assert not (args.only_train_primary and args.only_train_secondary), "cannot only train primary and also only train secondary model"
    print("Diffusion model created")

    # define callbacks
    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every, 
        save_top_k=-1, 
        dirpath=save_path
    )
    demo_callback = DemoCallback(config, args.save_path)

    # setup wandb logger
    if args.wandb_key is not None:
        wandb.login(key = args.wandb_key)
    wandb_logger = pl.loggers.WandbLogger(project='latent_dac_diffusion_audio_clap', log_model='all')
    wandb_logger.watch(diffusion_pl_module)
    # push_wandb_config(wandb_logger, args)

    # define pl training class
    if args.num_gpus > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    diffusion_trainer = pl.Trainer(
        devices = args.num_gpus,
        accelerator = accelerator,
        strategy = strategy,
        precision = 16,
        accumulate_grad_batches = 1, 
        callbacks = [ckpt_callback, demo_callback, exc_callback], # DEBUG
        logger = wandb_logger,
        log_every_n_steps = 1,
        val_check_interval = 1.0,
        max_epochs = config.max_epochs,
        profiler = "simple"
    )

    # start training
    diffusion_trainer.fit(diffusion_pl_module, data_module)

    # profile the time used to load data
    print("total training visits", data_module.dataset.total_visits)
    print("total train data loading time (sec)", data_module.dataset.total_runtime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC latent diffusion.')
    parser.add_argument(
        '-h5-dir', type=str,
        help='the audio h5 dataset path encoding dac and clap embeddings (multiple files) for training'
    )
    parser.add_argument(
        '--another-h5-dir', type=str, nargs='?',
        help='another audio h5 dataset path encoding dac and clap embeddings (multiple files) for training'
    )
    parser.add_argument(
        '--val-h5-dir', type=str, nargs='?',
        help='the audio h5 dataset path encoding dac and clap embeddings (multiple files) for val'
    )
    parser.add_argument(
        '--save-path', type=str, default='',
        help='the directory that model results and checkpoints are saved'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help='the number of gpus'
    )
    parser.add_argument(
        '--batch-size', type=int, nargs='?',
        help='batch size under use. If not specified, then use the config file'
    )
    parser.add_argument(
        '--frame-len-dac', type=int, nargs='?',
        help='the length of dac latents to generate. If not specified, then use the config file'
    )
    parser.add_argument(
        '--load-ckpt-path', type=str, nargs='?',
        help='the checkpoint path to load'
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
        '--prediction-type', type=str, default='sample',
        help='choose from "epsilon", "sample", "v_prediction"; default: "sample"'
    )
    parser.add_argument(
        '--only-train-primary', type=bool, default=False,
        help='If "VALL-E" pattern is chosen, decide whether or not to train only the first codebook model'
    )
    parser.add_argument(
        '--only-train-secondary', type=bool, default=False,
        help='If "VALL-E" pattern is chosen, decide whether or not to train only the rest codebooks model'
    )
    parser.add_argument(
        '--only-train-chroma', type=bool, default=False,
        help='If "VALL-E" pattern is chosen, decide whether or not to train only the chroma model'
    )
    parser.add_argument(
        '--no-clap-feat', type=bool, default=False,
        help='If True, then set the CLAP and meta features as None in each load'
    )
    parser.add_argument(
        '--dataset-size', type=int, nargs='?',
        help='the dataset size of random-loaded dataset, default: as indicated by config'
    )
    parser.add_argument(
        '--max-epochs', type=int, nargs='?',
        help='the max training epochs, default: as indicated by config'
    )
    parser.add_argument(
        '--scheduler', type=str, default='handcrafted',
        help='choose from "handcrafted", "diffusers"; default: "handcrafted"'
    )
    parser.add_argument(
        '--wandb-key', type=str, nargs='?',
        help='for login to wandb'
    )
    args = parser.parse_args()
    main(args)
