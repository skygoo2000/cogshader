#!/bin/bash

# CUDA 12.1 Environment Setup
export CUDA_HOME=/cm/shared/apps/cuda-12.1
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 


export CUDA_VISIBLE_DEVICES=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python tests/inference.py \
    --prompt <"prompt text"> \
    --model_path <model_path> \
    --tracking_path <tracking_path> \
    --image_or_video_path <image_or_video_path> \
    --generate_type i2v