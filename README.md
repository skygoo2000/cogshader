# Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control 🧪

<table align="center">
<tr>
  <td align="center"><video src="https://igl-hkust.github.io/das/static/videos/teaser.mp4">Your browser does not support the video tag.</video></td>
</tr>
</table>

## Quickstart

### Create environment
1. Clone the repository and create conda environment: 

```
git clone git@github.com:IGL-HKUST/DiffusionAsShader.git
conda create -n das python=3.10
conda activate das
```

2. Install pytorch, we recommend `Pytorch 2.5.1` with `CUDA 11.8`:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


<!-- 3. Install `MoGe`:
```
pip install git+https://github.com/asomoza/image_gen_aux.git
``` -->

3. Make sure the requirements are installed:
```
pip install -r requirements.txt
```

4. Manually download the SpatialTracker checkpoint to `checkpoints/`, from [Google Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ). 

<!-- 5. Manually download the ZoeDepth checkpoints (dpt_beit_large_384.pt, ZoeD_M12_K.pt, ZoeD_M12_NK.pt) to `models/monoD/zoeDepth/ckpts/`. For more information, refer to [this issue](https://github.com/henry123-boy/SpaTracker/issues/20). -->

<!-- Then download a dataset:

```bash
# install `huggingface_hub`
huggingface-cli download \
  --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset \
  --local-dir video-dataset-disney
``` -->

### Prepare Dataset

Before starting the training, please check whether the dataset has been prepared according to the [dataset specifications](assets/dataset.md). 

In short, your dataset structure should look like this. Running the `tree` command, you should see:

```
dataset
├── prompt.txt
├── videos.txt
├── trackings.txt
├── images.txt (or repaint.txt)

├── images (or repaint)
    ├── images/00000.png
    ├── images/00001.png
    ├── ...

├── tracking
    ├── tracking/00000_tracking.mp4
    ├── tracking/00001_tracking.mp4
    ├── ...

├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...

```

### Inference

We provide a script for inference. You can use the `infer.sh` script to generate videos.
Or run the `inference.py` script directly.

```python
python tests/inference.py \
    --prompt <"prompt text"> \ # prompt text
    --model_path <model_path> \ # checkpoint path
    --tracking_path <tracking_path> \ # the list of tracking videos path
    --image_or_video_path <image_or_video_path> \ # the list of images or videos path
    --generate_type i2v \ 
```

### Training

We provide training scripts suitable for image-to-video generation, compatible with the [CogVideoX model family](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce). Training can be started using the `train*.sh` scripts, depending on the task you want to train.

- Configure environment variables as per your choice:

  ```bash
  export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export WANDB_MODE="offline"
  export NCCL_P2P_DISABLE=1
  export TORCH_NCCL_ENABLE_MONITORING=0
  ```

- Configure which GPUs to use for training: `GPU_IDS="0,1"`

- Choose hyperparameters for training. Let's try to do a sweep on learning rate and optimizer type as an example:

  ```bash
  LEARNING_RATES=("1e-4" "1e-3")
  LR_SCHEDULES=("cosine_with_restarts")
  OPTIMIZERS=("adamw" "adam")
  MAX_TRAIN_STEPS=("2000")
  ```

- Select which Accelerate configuration you would like to train with: `ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_1.yaml"`. We provide some default configurations in the `accelerate_configs/` directory - single GPU uncompiled/compiled, 2x GPU DDP, DeepSpeed, etc. You can create your own config files with custom settings using `accelerate config --config_file my_config.yaml`.

- Specify the absolute paths and columns/files for captions and videos.

  ```bash
  DATA_ROOT="../datasets/cogshader"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  TRACKING_COLUMN="trackings.txt"
  ```

- Launch experiments sweeping different hyperparameters:
  ```
  # training dataset parameters
  DATA_ROOT="../datasets/cogshader"
  MODEL_PATH="../ckpts/CogVideoX-5b-I2V"
  CAPTION_COLUMN="prompt.txt"
  VIDEO_COLUMN="videos.txt"
  TRACKING_COLUMN="trackings.txt"

  # validation parameters
  TRACKING_MAP_PATH="../eval/3d/tracking/dance_tracking.mp4"
  VALIDATION_PROMPT="text"
  VALIDATION_IMAGES="../000000046_0.png"

  for learning_rate in "${LEARNING_RATES[@]}"; do
    for lr_schedule in "${LR_SCHEDULES[@]}"; do
      for optimizer in "${OPTIMIZERS[@]}"; do
        for steps in "${MAX_TRAIN_STEPS[@]}"; do
          output_dir="/aifs4su/mmcode/lipeng/cogvideo/ckpts/cogshader_inv-avatar-physics_steps_${steps}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

          cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS --num_processes $NUM_PROCESSES --main_process_port $PORT training/cogvideox_image_to_video_sft.py \
            --pretrained_model_name_or_path $MODEL_PATH \
            --data_root $DATA_ROOT \
            --caption_column $CAPTION_COLUMN \
            --video_column $VIDEO_COLUMN \
            --tracking_column $TRACKING_COLUMN \
            --tracking_map_path $TRACKING_MAP_PATH \
            --num_tracking_blocks 18 \
            --height_buckets 480 \
            --width_buckets 720 \
            --frame_buckets 49 \
            --dataloader_num_workers 8 \
            --pin_memory \
            --validation_prompt $VALIDATION_PROMPT \
            --validation_images $VALIDATION_IMAGES \
            --validation_prompt_separator ::: \
            --num_validation_videos 1 \
            --validation_epochs 1 \
            --seed 42 \
            --mixed_precision bf16 \
            --output_dir $output_dir \
            --max_num_frames 49 \
            --train_batch_size $TRAIN_BATCH_SIZE \
            --max_train_steps $steps \
            --checkpointing_steps $CHECKPOINT_STEPS \
            --gradient_accumulation_steps 4 \
            --gradient_checkpointing \
            --learning_rate $learning_rate \
            --lr_scheduler $lr_schedule \
            --lr_warmup_steps $WARMUP_STEPS \
            --lr_num_cycles 1 \
            --enable_slicing \
            --enable_tiling \
            --optimizer $optimizer \
            --beta1 0.9 \
            --beta2 0.95 \
            --weight_decay 0.001 \
            --noised_image_dropout 0.05 \
            --max_grad_norm 1.0 \
            --allow_tf32 \
            --report_to wandb \
            --resume_from_checkpoint \"latest\" \
            --nccl_timeout 1800"
          
          echo "Running command: $cmd"
          eval $cmd
          echo -ne "-------------------- Finished executing script --------------------\n\n"
        done
      done
    done
  done
  ```

  To understand what the different parameters mean, you could either take a look at the [args](./training/args.py) file or run the training script with `--help`.



