#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/cogvideo/lib:$HOME/miniconda3/envs/cogvideo/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_VISIBLE_DEVICES=6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python tests/inference.py \
    --prompt "A group of people are gathered at an outdoor car show and they are admiring various classic cars." \
    --model_path /aifs4su/mmcode/lipeng/cogvideo/ckpts/8000_img_cfg_cogvideox-sft__optimizer_adamw__steps_3000__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2800 \
    --tracking_path /aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira/tracking/000000046_0_tracking.mp4 \
    --image_or_video_path /aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira200/000000046_0.png \
    --generate_type i2v
