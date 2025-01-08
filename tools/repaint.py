import torch
import os
import cv2
import argparse
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

def extract_first_frame(video_path, output_path):
    """提取视频第一帧"""
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    if success:
        cv2.imwrite(output_path, frame)
        print(f"已保存第一帧到 {output_path}")
    else:
        print(f"无法读取视频: {video_path}")
    video.release()
    return success

def process_video_directory(input_dir, firstframe_dir):
    """处理视频目录，提取所有视频的第一帧"""
    os.makedirs(firstframe_dir, exist_ok=True)
    processed_files = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, filename)
            frame_path = os.path.join(firstframe_dir, f"{os.path.splitext(filename)[0]}.png")
            if extract_first_frame(video_path, frame_path):
                processed_files.append(os.path.splitext(filename)[0])
    
    return processed_files

def setup_model(gpu_id):
    """设置和初始化模型"""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev", 
        torch_dtype=torch.bfloat16
    ).to(device)
    processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
    
    return pipe, processor, device

def repaint_images(pipe, processor, device, dataset_dir, firstframe_dir, processed_files):
    """对提取的帧进行重绘"""
    # 读取prompt文件
    with open(os.path.join(dataset_dir, 'prompt.txt'), 'r') as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts]
    
    # 创建重绘输出目录
    repaint_dir = os.path.join(dataset_dir, 'repaint')
    os.makedirs(repaint_dir, exist_ok=True)
    
    # 处理每个提取的帧
    for i, (filename, prompt) in enumerate(zip(processed_files, prompts)):
        print(f"正在重绘图片 {i+1}/{len(processed_files)}: {filename}")
        
        # 加载控制图片
        control_image = load_image(os.path.join(firstframe_dir, f"{filename}.png"))
        control_image = processor(control_image)[0].convert("RGB")
        
        # 生成图片
        image = pipe(
            prompt=prompt,
            control_image=control_image,
            height=480,
            width=720,
            num_inference_steps=30,
            guidance_scale=10.0,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images[0]
        
        # 保存输出图片
        output_path = os.path.join(repaint_dir, f"{filename}.png")
        image.save(output_path)
        print(f"已保存重绘结果到 {output_path}")

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集目录路径')
    args = parser.parse_args()
    
    # 设置路径
    videos_dir = os.path.join(args.dataset_dir, 'videos')
    firstframe_dir = os.path.join(args.dataset_dir, 'firstframe')
    os.makedirs(firstframe_dir, exist_ok=True)
    
    # 1. 提取视频第一帧
    print("第一步：提取视频第一帧")
    processed_files = process_video_directory(videos_dir, firstframe_dir)
    
    if not processed_files:
        print("没有找到可处理的视频文件")
        return
    
    # 2. 设置模型
    print("第二步：初始化模型")
    pipe, processor, device = setup_model(args.gpu)
    
    # 3. 重绘图片
    print("第三步：开始重绘图片")
    repaint_images(pipe, processor, device, args.dataset_dir, firstframe_dir, processed_files)
    
    print("所有处理完成！")

if __name__ == "__main__":
    main()