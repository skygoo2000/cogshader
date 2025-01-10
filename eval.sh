#!/bin/bash

# CUDA 12.1 Environment Setup
export CUDA_HOME=/cm/shared/apps/cuda-12.1
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} 

export CUDA_VISIBLE_DEVICES=2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvideo

echo "start time: $(date)"

cd /home/lipeng/cogvideox-finetune

python tests/evaluation.py \
    --data_root /aifs4su/mmcode/lipeng/cogvideo/eval/static_eval \
    --model_path /aifs4su/mmcode/lipeng/cogvideo/ckpts/cogshader_inv-avatar-physics_steps_2000__optimizer_adamw__lr-schedule_cosine_with_restarts__learning-rate_1e-4/checkpoint-2000 \
    --evaluation_dir coginv_scene_eval \
    --fps 12 \
    --generate_type i2v \
    --tracking_column trackings.txt \
    --video_column videos.txt \
    --caption_column prompt.txt \
    # --image_paths repaint.txt