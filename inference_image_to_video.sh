#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/cogvideo/lib:$HOME/miniconda3/envs/cogvideo/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_VISIBLE_DEVICES=6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python tests/inference.py \
    --prompt <"prompt text"> \
    --model_path <model_path> \
    --tracking_path <tracking_path> \
    --image_or_video_path <image_or_video_path> \
    --generate_type i2v