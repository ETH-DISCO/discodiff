# DiscoDiff | [📜Paper](https://ieeexplore.ieee.org/document/10888056) | [🤗Captions](https://huggingface.co/datasets/disco-eth/jamendo-fma-captions) | [🤗Ranking](https://huggingface.co/datasets/disco-eth/FMA-rank)

A text-to-music diffusion model. We also provide high-quality synthetic captions for MTG-Jamendo and FMA, as well as a ranking for every song in the FMA dataset.

## Installation

```conda create -n discodiff python=3.11```

```conda activate discodiff```

```pip install requirements.txt```

```conda env update --file environment.yml```

## Inference

```
python -u inference_conditional.py \
--save-dir ./results/inference \
--load-ckpt-path checkpoint/path \
--text-prompt "Techno music with heavy beats." \
--dur-sec 10.0
```

Checkpoint to be released.

## Training and Testing

### Dataset Preparation

We first process audio files into hdf5 files for faster dataloading during training. Therefore, preprocessing is needed.

Download MTD-Jamendo raw_30s dataset from https://mtg.github.io/mtg-jamendo-dataset/ and/or download FMA-large/FMA-full dataset from https://github.com/mdeff/fma

Extract the files in each dataset, a folder containing multiple numbered folders (each numbered folder contains audio files) will be created.

Then run

```
python -u data/create_h5_dataset_multifolder.py \
--root-dir the/dir/containing/folders/of/audio/files \
--target-dir the/dir/outputting/hdf5/dataset/files \
--json-path ./data/json_metadata/one_of_the_json_file
```

### Training

```
python -u train_conditional.py \
-h5-dir the/hdf5/training/dataset/folder \
--another-h5-dir [optional] we/support/loading/another/optional/hdf5/dataset \
--save-path [optional] the/dir/where/val/results/and/ckpts/are/saved \
--num-gpus 8 \
--load-ckpt-path [optional] ckpt/path/to/resume \
--rvq-pattern VALL-E \
--model-size large \
--prediction-type v_prediction \
--val-h5-dir the/hdf5/validation/dataset/folder \
--batch-size 256 \
--frame-len-dac 2320 \
--scheduler handcrafted \
--wandb-key your_wandb_key_for_logging
```

There is another option to train primary/secondary/chroma diffusion model only in the training script. If following this way, you may need merging the checkpoints with

```
python merge_ckpts.py \
-primary-ckpt-path ./results/one_exp/epoch=142-step=114000.ckpt \
-secondary-ckpt-path ./results/one_exp/epoch=139-step=112000.ckpt \
-chroma-ckpt-path ./results/one_exp/epoch=199-step=10000.ckpt \
-out-ckpt-path ./results/merged_ckpts/merged.ckpt \
--rvq-pattern VALL-E --model-size large
```

### Testing

```
python -u test_conditional.py \
-h5-dir the/hdf5/testing/dataset/folder \
--save-path  ./results/test_results \
--load-ckpt-path  ckpt/path/to/load \
--frame-len-dac 2320 \
--rvq-pattern  VALL-E \
--model-size large \
--prediction-type v_prediction \
--scheduler handcrafted \
--wandb-key your_wandb_key_for_logging
```

