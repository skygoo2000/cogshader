#!/bin/bash

# CUDA 12.1 Environment Setup
export CUDA_HOME=/cm/shared/apps/cuda-12.1
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate das

cd /your_path

export DATASET_PATH=../your_dataset_path

mkdir $DATASET_PATH/generated

export CUDA_VISIBLE_DEVICES=6,7
PORT=29501
accelerate launch --main_process_port $PORT accelerate_tracking.py --root $DATASET_PATH/videos --outdir $DATASET_PATH/generated --grid_size 70 
# python batch_tracking.py --root $DATASET_PATH/videos --outdir $DATASET_PATH/generated --grid_size 70 