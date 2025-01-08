"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
from typing import Any, Dict, List, Literal, Tuple
import pandas as pd
import os
import sys

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video

import numpy as np
import random
import cv2
from pathlib import Path
import decord
from torchvision import transforms
from torchvision.transforms.functional import resize

import PIL.Image
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from models.cogvideox_tracking import CogVideoXImageToVideoPipelineTracking, CogVideoXPipelineTracking, CogVideoXVideoToVideoPipelineTracking
from training.dataset import VideoDataset, VideoDatasetWithResizingTracking

class VideoDatasetWithResizingTrackingEval(VideoDataset):
    def __init__(self, *args, **kwargs) -> None:
        self.tracking_column = kwargs.pop("tracking_column", None)
        self.image_paths = kwargs.pop("image_paths", None)
        super().__init__(*args, **kwargs)

    def _preprocess_video(self, path: Path, tracking_path: Path, image_paths: Path = None) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path, tracking_path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            # 读取图像 - 使用PIL而不是decord
            image = Image.open(image_paths)
            # 转换为RGB模式（如果是其他模式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # 转换为tensor
            image = torch.from_numpy(np.array(image)).float()
            image = image.permute(2, 0, 1).contiguous()
            image = resize(image, nearest_res)
            image = self.video_transforms(image)

            tracking_reader = decord.VideoReader(uri=tracking_path.as_posix())
            tracking_frames = tracking_reader.get_batch(frame_indices)
            tracking_frames = tracking_frames[:nearest_frame_bucket].float()
            tracking_frames = tracking_frames.permute(0, 3, 1, 2).contiguous()
            tracking_frames_resized = torch.stack([resize(tracking_frame, nearest_res) for tracking_frame in tracking_frames], dim=0)
            tracking_frames = torch.stack([self.video_transforms(tracking_frame) for tracking_frame in tracking_frames_resized], dim=0)

            return image, frames, tracking_frames, None

    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]
    
    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        tracking_path = self.data_root.joinpath(self.tracking_column)
        image_paths = self.data_root.joinpath(self.image_paths)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not tracking_path.exists() or not tracking_path.is_file():
            raise ValueError(
                "Expected `--tracking_column` to be path to a file in `--data_root` containing line-separated tracking information."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(tracking_path, "r", encoding="utf-8") as file:
            tracking_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        with open(image_paths, "r", encoding="utf-8") as file:
            image_paths_list = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        self.image_paths = image_paths_list
        return prompts, video_paths

    def _load_dataset_from_csv(self) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(self.dataset_file)
        prompts = df[self.caption_column].tolist()
        video_paths = df[self.video_column].tolist()
        tracking_paths = df[self.tracking_column].tolist()
        image_paths = df[self.image_paths].tolist()
        video_paths = [self.data_root.joinpath(line.strip()) for line in video_paths]
        tracking_paths = [self.data_root.joinpath(line.strip()) for line in tracking_paths]
        image_paths = [self.data_root.joinpath(line.strip()) for line in image_paths]

        if any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found at least one path that is not a valid file."
            )

        self.tracking_paths = tracking_paths
        self.image_paths = image_paths
        return prompts, video_paths
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        if self.load_tensors:
            image_latents, video_latents, prompt_embeds = self._preprocess_video(self.video_paths[index], self.tracking_paths[index])

            # The VAE's temporal compression ratio is 4.
            # The VAE's spatial compression ratio is 8.
            latent_num_frames = video_latents.size(1)
            if latent_num_frames % 2 == 0:
                num_frames = latent_num_frames * 4
            else:
                num_frames = (latent_num_frames - 1) * 4 + 1

            height = video_latents.size(2) * 8
            width = video_latents.size(3) * 8

            return {
                "prompt": prompt_embeds,
                "image": image_latents,
                "video": video_latents,
                "tracking_map": tracking_map,
                "video_metadata": {
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                },
            }
        else:
            image, video, tracking_map, _ = self._preprocess_video(self.video_paths[index], self.tracking_paths[index], self.image_paths[index])

            return {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "tracking_map": tracking_map,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                },
            }
    
    def _load_preprocessed_latents_and_embeds(self, path: Path, tracking_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        # The current path is something like: /a/b/c/d/videos/00001.mp4
        # We need to reach: /a/b/c/d/video_latents/00001.pt
        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        tracking_map_path = path.parent.parent.joinpath("tracking_map")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or not tracking_map_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains folders named `video_latents`, `prompt_embeds`, and `tracking_map`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        tracking_map_filepath = tracking_map_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file() or not tracking_map_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            tracking_map_filepath = tracking_map_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} or {tracking_map_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        tracking_map = torch.load(tracking_map_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, tracking_map, embeds

def sample_from_dataset(
    data_root: str,
    caption_column: str,
    tracking_column: str,
    image_paths: str,
    video_column: str,
    num_samples: int = -1,
    random_seed: int = 42
):
    """从数据集中抽取样本"""
    if image_paths:
        # 如果提供了image_paths，使用VideoDatasetWithResizingTrackingEval
        dataset = VideoDatasetWithResizingTrackingEval(
            data_root=data_root,
            caption_column=caption_column,
            tracking_column=tracking_column,
            image_paths=image_paths,
            video_column=video_column,
            max_num_frames=49,
            load_tensors=False,
            random_flip=None,
            frame_buckets=[49],
            image_to_video=True
        )
    else:
        # 如果没有提供image_paths，使用VideoDatasetWithResizingTracking
        dataset = VideoDatasetWithResizingTracking(
            data_root=data_root,
            caption_column=caption_column,
            tracking_column=tracking_column,
            video_column=video_column,
            max_num_frames=49,
            load_tensors=False,
            random_flip=None,
            frame_buckets=[49],
            image_to_video=True
        )
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 随机抽取样本
    total_samples = len(dataset)
    if num_samples == -1:
        # 如果num_samples为-1，处理所有样本
        selected_indices = range(total_samples)
    else:
        selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    samples = []
    for idx in selected_indices:
        sample = dataset[idx]
        # 根据dataset.__getitem__的返回值获取数据
        image = sample["image"]  # 已经是处理好的tensor
        video = sample["video"]  # 已经是处理好的tensor
        tracking_map = sample["tracking_map"]  # 已经是处理好的tensor
        prompt = sample["prompt"]
        
        samples.append({
            "prompt": prompt,
            "tracking_frame": tracking_map[0],  # 取第一帧
            "video_frame": image,  # 取第一帧
            "video": video,  # 完整的video
            "tracking_maps": tracking_map,  # 完整的tracking maps
            "height": sample["video_metadata"]["height"],
            "width": sample["video_metadata"]["width"]
        })
    
    return samples

def generate_video(
    prompt: str,
    model_path: str,
    tracking_path: str = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["i2v", "i2vo", "v2v"],
    seed: int = 42,
    data_root: str = None,
    caption_column: str = None,
    tracking_column: str = None,
    video_column: str = None,
    image_paths: str = None,
    num_samples: int = -1,
    evaluation_dir: str = "evaluations",
    fps: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # yr: get all video names first
    video_names = []
    video_txt = os.path.join(data_root, video_column)
    with open(video_txt , "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # 去掉首尾空白字符
            name = line.split('/')[-1].split('.')[0]  # 按 '/' 分割取最后一部分，再按 '.' 分割取第一个
            video_names.append(name)

    # 如果提供了数据集参数，从数据集中采样
    samples = None
    if all([data_root, caption_column, tracking_column, video_column]):
        samples = sample_from_dataset(
            data_root=data_root,
            caption_column=caption_column,
            tracking_column=tracking_column,
            image_paths=image_paths,
            video_column=video_column,
            num_samples=num_samples,
            random_seed=seed
        )
    
    # 加载模型和数据
    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipelineTracking.from_pretrained(model_path, torch_dtype=dtype)
        if not samples:
            image = load_image(image=image_or_video_path)
            height, width = image.height, image.width
    elif generate_type == "i2vo":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=dtype)
        if not samples:
            image = load_image(image=image_or_video_path)
            height, width = image.height, image.width
    else:  # v2v
        if tracking_column:
            pipe = CogVideoXVideoToVideoPipelineTracking.from_pretrained(model_path, torch_dtype=dtype)
        else:
            pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        if not samples:
            video = load_video(image_or_video_path)

    # 设置模型参数
    pipe.to(device, dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.transformer.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    pipe.transformer.gradient_checkpointing = False
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 生成视频

    if samples:
        from tqdm import tqdm
        for i, sample in tqdm(enumerate(samples), desc="处理样本"):
            print(f"当前prompt: {sample['prompt'][:30]}")
            tracking_frame = sample["tracking_frame"].to(device=device, dtype=dtype)
            video_frame = sample["video_frame"].to(device=device, dtype=dtype)
            video = sample["video"].to(device=device, dtype=dtype)
            tracking_maps = sample["tracking_maps"].to(device=device, dtype=dtype)
            
            # VAE编码tracking maps
            print("encoding tracking maps")
            tracking_video = tracking_maps
            tracking_maps = tracking_maps.unsqueeze(0)
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            with torch.no_grad():
                tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
                tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
                tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]


            # 根据不同生成类型设置参数
            pipeline_args = {
                "prompt": sample["prompt"],
                "negative_prompt": "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                "num_inference_steps": num_inference_steps,
                "num_frames": 49,
                "use_dynamic_cfg": True,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator(device=device).manual_seed(seed),
                "height": sample["height"],
                "width": sample["width"]
            }

            if generate_type == "i2v" or generate_type == "i2vo":
                pipeline_args["image"] = (video_frame + 1.0) / 2.0
            else:  # v2v
                pipeline_args["video"] = video_frame

            if tracking_column and generate_type != "i2vo":
                pipeline_args["tracking_maps"] = tracking_maps
                pipeline_args["tracking_image"] = (tracking_frame.unsqueeze(0) + 1.0) / 2.0

            with torch.no_grad():
                video_generate = pipe(**pipeline_args).frames[0]

            # 修改输出路径格式
            output_dir = os.path.join(data_root, evaluation_dir)
            # output_name = f"{i:04d}.mp4"
            output_name = f'{video_names[i]}.mp4'
            output_file = os.path.join(output_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            export_concat_video(video_generate, video, tracking_video, output_file, video_name=video_names[i], fps=fps)
            
    else:
        # 处理单个视频生成
        pipeline_args = {
            "prompt": prompt,
            "num_videos_per_prompt": num_videos_per_prompt,
            "num_inference_steps": num_inference_steps,
            "num_frames": 49,
            "use_dynamic_cfg": True,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator().manual_seed(seed),
        }

        pipeline_args["video"] = video

        if generate_type == "i2v" or generate_type == "i2vo":
            pipeline_args["image"] = image
            pipeline_args["height"] = height
            pipeline_args["width"] = width

        if tracking_path and generate_type != "i2vo":
            tracking_maps = load_video(tracking_path)
            tracking_maps = torch.stack([
                torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0 
                for frame in tracking_maps
            ]).to(device=device, dtype=dtype)
            
            tracking_video = tracking_maps
            tracking_maps = tracking_maps.unsqueeze(0)
            tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
            with torch.no_grad():
                tracking_latent_dist = pipe.vae.encode(tracking_maps).latent_dist
                tracking_maps = tracking_latent_dist.sample() * pipe.vae.config.scaling_factor
                tracking_maps = tracking_maps.permute(0, 2, 1, 3, 4)
            
            pipeline_args["tracking_maps"] = tracking_maps
            pipeline_args["tracking_image"] = tracking_maps[:, :1]

        with torch.no_grad():
            video_generate = pipe(**pipeline_args).frames[0]

        # 单个视频生成的输出路径
        output_dir = os.path.join(data_root, evaluation_dir)
        output_name = f"{os.path.splitext(os.path.basename(image_or_video_path))[0]}.mp4"
        output_file = os.path.join(output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)
        export_concat_video(video_generate, video, tracking_video, output_file, video_names=None, fps=fps)

def create_frame_grid(frames: List[np.ndarray], interval: int = 9, max_cols: int = 7) -> np.ndarray:
    """
    将视频帧按间隔采样并横向排列成网格图片
    
    Args:
        frames: 视频帧列表
        interval: 采样间隔
        max_cols: 每行最多显示的帧数
    
    Returns:
        网格化的图片数组
    """
    # 按间隔采样帧
    sampled_frames = frames[::interval]
    
    # 计算行数和列数
    n_frames = len(sampled_frames)
    n_cols = min(max_cols, n_frames)
    n_rows = (n_frames + n_cols - 1) // n_cols
    
    # 获取单帧的高度和宽度
    frame_height, frame_width = sampled_frames[0].shape[:2]
    
    # 创建空白画布
    grid = np.zeros((frame_height * n_rows, frame_width * n_cols, 3), dtype=np.uint8)
    
    # 填充帧
    for idx, frame in enumerate(sampled_frames):
        i = idx // n_cols
        j = idx % n_cols
        grid[i*frame_height:(i+1)*frame_height, j*frame_width:(j+1)*frame_width] = frame
    
    return grid

def export_concat_video(
    generated_frames: List[PIL.Image.Image], 
    original_video: torch.Tensor,
    tracking_maps: torch.Tensor = None,
    output_video_path: str = None,
    video_name: str = None,
    fps: int = 8
) -> str:
    """
    将生成的视频帧、原始视频和tracking maps导出为视频文件，并将采样的帧分别保存到不同文件夹
    """
    import imageio
    import os
    
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
        
    # 创建子文件夹
    base_dir = os.path.dirname(output_video_path)
    generated_dir = os.path.join(base_dir, "generated")  # 用于存放生成的视频
    group_dir = os.path.join(base_dir, "group")  # 用于存放拼接的视频
    
    # 获取文件名（不含路径）并创建该视频的专属文件夹
    filename = os.path.basename(output_video_path)
    if video_name:
        name_without_ext = video_name
    else:
        name_without_ext = os.path.splitext(filename)[0]
    video_frames_dir = os.path.join(base_dir, "frames", name_without_ext)  # frames/video_name/
    
    # 在视频专属文件夹下创建三个子文件夹
    groundtruth_dir = os.path.join(video_frames_dir, "gt")
    generated_frames_dir = os.path.join(video_frames_dir, "generated")
    tracking_dir = os.path.join(video_frames_dir, "tracking")
    
    # 创建所需的所有目录
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(group_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)
    os.makedirs(generated_frames_dir, exist_ok=True)
    os.makedirs(tracking_dir, exist_ok=True)
    
    # 将原始视频张量转换为numpy数组并调整为正确的格式
    original_frames = []
    for frame in original_video:
        frame = frame.permute(1,2,0).to(dtype=torch.float32,device="cpu").numpy()
        frame = ((frame + 1.0) * 127.5).astype(np.uint8)
        original_frames.append(frame)
    
    tracking_frames = []
    if tracking_maps is not None:
        for frame in tracking_maps:
            frame = frame.permute(1,2,0).to(dtype=torch.float32,device="cpu").numpy()
            frame = ((frame + 1.0) * 127.5).astype(np.uint8)
            tracking_frames.append(frame)
    
    # 确保所有视频的帧数相同
    num_frames = min(len(generated_frames), len(original_frames))
    if tracking_maps is not None:
        num_frames = min(num_frames, len(tracking_frames))
    
    generated_frames = generated_frames[:num_frames]
    original_frames = original_frames[:num_frames]
    if tracking_maps is not None:
        tracking_frames = tracking_frames[:num_frames]
    
    # 将生成的PIL图像转换为numpy数组
    generated_frames_np = [np.array(frame) for frame in generated_frames]
    
    # 单独保存生成的视频到generated文件夹
    gen_video_path = os.path.join(generated_dir, f"{name_without_ext}.mp4")
    with imageio.get_writer(gen_video_path, fps=fps) as writer:
        for frame in generated_frames_np:
            writer.append_data(frame)
    
    # 上下拼接每一帧并保存采样帧
    concat_frames = []
    for i in range(num_frames):
        gen_frame = generated_frames_np[i]
        orig_frame = original_frames[i]
        
        # 确保所有帧的宽度相同
        width = min(gen_frame.shape[1], orig_frame.shape[1])
        height = orig_frame.shape[0]  # 使用原始视频的高度作为标准
        
        # 调整所有帧的大小
        gen_frame = Image.fromarray(gen_frame).resize((width, height))
        gen_frame = np.array(gen_frame)
        orig_frame = Image.fromarray(orig_frame).resize((width, height))
        orig_frame = np.array(orig_frame)
        
        if tracking_maps is not None:
            track_frame = tracking_frames[i]
            track_frame = Image.fromarray(track_frame).resize((width, height))
            track_frame = np.array(track_frame)
            # 三个视频垂直拼接
            concat_frame = np.concatenate([gen_frame, orig_frame, track_frame], axis=0)
        else:
            # 视频垂直拼接
            concat_frame = np.concatenate([gen_frame, orig_frame], axis=0)
        
        concat_frames.append(concat_frame)
        
        # 每9帧保存一次各类型的帧
        if i % 9 == 0:
            # 保存生成的帧
            gen_frame_path = os.path.join(generated_frames_dir, f"{i:04d}.png")
            Image.fromarray(gen_frame).save(gen_frame_path)
            
            # 保存原始帧
            gt_frame_path = os.path.join(groundtruth_dir, f"{i:04d}.png")
            Image.fromarray(orig_frame).save(gt_frame_path)
            
            # 如果有tracking maps，保存tracking帧
            if tracking_maps is not None:
                track_frame_path = os.path.join(tracking_dir, f"{i:04d}.png")
                Image.fromarray(track_frame).save(track_frame_path)
    
    # 导出拼接后的视频到group文件夹
    group_video_path = os.path.join(group_dir, filename)
    with imageio.get_writer(group_video_path, fps=fps) as writer:
        for frame in concat_frames:
            writer.append_data(frame)
            
    return group_video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="Optional: override the prompt from dataset")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="i2v", help="The type of video generation (e.g., 'i2v', 'i2vo')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--tracking_path", type=str, default=None, help="The path of the tracking maps to be used")
    
    # 数据集相关参数设为必需
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--caption_column", type=str, required=True, help="Name of the caption column")
    parser.add_argument("--tracking_column", type=str, required=True, help="Name of the tracking column")
    parser.add_argument("--video_column", type=str, required=True, help="Name of the video column")
    parser.add_argument("--image_paths", type=str, required=False, help="Name of the image column")
    
    # 添加num_samples参数
    parser.add_argument("--num_samples", type=int, default=-1, 
                       help="Number of samples to process. -1 means process all samples")
    
    # 添加evaluation_dir参数
    parser.add_argument("--evaluation_dir", type=str, default="evaluations", 
                       help="Name of the directory to store evaluation results")
    
    # 添加fps参数
    parser.add_argument("--fps", type=int, default=8, 
                       help="Frames per second for the output video")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    # 如果没有提供prompt，generate_video函数会使用数据集中的prompt
    generate_video(
        prompt=args.prompt,  # 可以为None
        model_path=args.model_path,
        tracking_path=args.tracking_path,
        image_paths=args.image_paths,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        data_root=args.data_root,
        caption_column=args.caption_column,
        tracking_column=args.tracking_column,
        video_column=args.video_column,
        num_samples=args.num_samples,
        evaluation_dir=args.evaluation_dir,
        fps=args.fps,
    )