#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/cogvideo/lib:$HOME/miniconda3/envs/cogvideo/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_VISIBLE_DEVICES=7

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python tests/inference.py \
    --prompt "A group of people are gathered at an outdoor car show and they are admiring various classic cars. An open blue, grassy field surrounded by trees and residential buildings." \
    --model_path /home/lipeng/cogvideox-finetune/ckpts/img_cogvideox-sft__optimizer_adamw__steps_800__lr-schedule_cosine_with_restarts__learning-rate_1e-4 \
    --tracking_path /home/lipeng/cogvideox-finetune/datasets/cogmira/tracking/000000046_0_tracking.mp4 \
    --image_or_video_path /home/lipeng/cogvideox-finetune/datasets/cogmira/000000046_0_test.png \
    --generate_type i2v