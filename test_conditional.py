import sys
import os
import argparse

import pytorch_lightning as pl
import wandb

from train_conditional import sel_config, set_save_path, set_device_accelerator, config_adjustments
from pl_module_callbacks import LatentDiffusionCondModule, DacCLAPDataModule
from pl_module_callbacks import ExceptionCallback, DemoCallback

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
        frame_len_dac = args.frame_len_dac
    )

    # get dataset
    data_module = DacCLAPDataModule(
        h5_dir = args.h5_dir,
        config = config,
        val_h5_dir = args.h5_dir,
    )
    print("Dataset created")

    # get pl diffusion module (model defined inside)
    diffusion_pl_module = LatentDiffusionCondModule(
        config,
        ckpt_path = args.load_ckpt_path,
    )
    print("Diffusion model created")

    # define callbacks
    exc_callback = ExceptionCallback()
    demo_callback = DemoCallback(config, args.save_path)

    # setup wandb logger
    if args.wandb_key is not None:
        wandb.login(key = args.wandb_key)
    wandb_logger = pl.loggers.WandbLogger(project='test_latent_dac_diffusion_text_clap', log_model='all')
    wandb_logger.watch(diffusion_pl_module)
    # push_wandb_config(wandb_logger, args)

    # define pl training class
    if args.num_gpus > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    # define pl training class
    diffusion_trainer = pl.Trainer(
        devices = args.num_gpus,
        accelerator = accelerator,
        strategy = strategy,
        precision = 16,
        accumulate_grad_batches = 1, 
        callbacks = [demo_callback, exc_callback], # DEBUG
        logger = wandb_logger,
        log_every_n_steps = 1,
        val_check_interval = 1.0,
        max_epochs = config.max_epochs,
    )

    # start training
    diffusion_trainer.test(diffusion_pl_module, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training DAC RVQ diffusion.')
    parser.add_argument(
        '-h5-dir', type=str,
        help='the audio h5 dataset path encoding dac and clap embeddings (multiple files)'
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
        '--batch-size', type=int, default=1,
    )
    parser.add_argument(
        '--load-ckpt-path', type=str, nargs='?',
        help='the checkpoint path to load'
    )
    parser.add_argument(
        '--frame-len-dac', type=int, nargs='?',
        help='the length of dac latents to generate. If not specified, then use the config file'
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
        '--prediction-type', type=str, default='v_prediction',
        help='choose from "epsilon", "sample", "v_prediction"; default: "sample"'
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
