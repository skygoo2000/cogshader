#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/cogvideo/lib:$HOME/miniconda3/envs/cogvideo/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib
export CUDA_VISIBLE_DEVICES=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

# 打印启动时间
echo "程序开始启动时间: $(date)"

python tests/camctrl_eval.py \
    --data_root /aifs4su/mmcode/lipeng/cogvideo/eval/obj_rot \
    --model_path /aifs4su/mmcode/lipeng/cogvideo/ckpts/cogshader_inv-avatar-physics_steps_2000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000 \
    --generate_type i2v\
    --tracking_column trackings.txt \
    --video_column videos.txt \
    --caption_column prompt.txt \
    # --image_paths repaint.txt \

## 1. 修改dataset
## 2. 修改model path
## 4. 修改image_paths
## cogmira200: /aifs4su/mmcode/lipeng/cogvideo/ckpts/8000_img_cfg_cogvideox-sft__optimizer_adamw__steps_3000__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2800
## cogmira200+trackingavatars: /aifs4su/mmcode/lipeng/cogvideo/ckpts/200_img_cfg_cogvideox-sft__optimizer_adamw__steps_200__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-200
## depth: /aifs4su/mmcode/lipeng/cogvideo/ckpts/depth_cogmiramix__steps_1000_optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-1000
