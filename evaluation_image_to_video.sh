#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/cogvideo/lib:$HOME/miniconda3/envs/cogvideo/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_VISIBLE_DEVICES=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

python tests/evaluation.py \
    --data_root /aifs4su/mmcode/lipeng/cogvideo/datasets/3d \
    --model_path /aifs4su/mmcode/lipeng/cogvideo/ckpts/200_img_cfg_cogvideox-sft__optimizer_adamw__steps_200__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-200 \
    --generate_type i2v\
    --image_paths images.txt \
    --tracking_column trackings.txt \
    --video_column videos.txt \
    --caption_column prompt.txt