#!/bin/bash

# CUDA 12.1 Environment Setup
export CUDA_HOME=/cm/shared/apps/cuda-12.1
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 

export CUDA_VISIBLE_DEVICES=3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

echo "start time: $(date)"

python tests/evaluation.py \
    --data_root <data_root> \
    --model_path <model_path> \
    --evaluation_dir <evaluation_dir> \
    --fps 8 \
    --generate_type i2v \
    --tracking_column trackings.txt \
    --video_column videos.txt \
    --caption_column prompt.txt \
    --image_paths repaint.txt \