import os, sys
import numpy as np
import torch
import argparse
import cv2
import PIL
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from PIL import Image
from tools.gen_prompt import gen_prompt_for_rgb
from tools.depth_pro_tool import Depth_pro

def img2video(image_path, output_path, frame_nums=49):
     # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    height, width = img.shape[:2]

    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))

    for i in range(frame_nums):
        # 写入单帧
        out.write(img)

    out.release()

def crop_rgb(rgb, target_h=480, target_w=720, save_path=None):
    H, W, _ = rgb.shape

    if H != target_h and W != target_w:
    
        print(f"Original size: {H}x{W}")
        # first resize by scale
        scale = max(target_w / W, target_h / H)
        new_w = int(W * scale)
        new_h = int(H * scale)
        rgb = cv2.resize(rgb, (new_w, new_h))
        # then crop from center
        crop_top = (new_h - target_h) // 2
        crop_left = (new_w - target_w) // 2
        crop_bottom = crop_top + target_h
        crop_right = crop_left + target_w
        rgb = rgb[crop_top:crop_bottom, crop_left:crop_right]
        print(f"Cropped size: {rgb.shape}")
    # save the cropped image if needed
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return rgb

# return frames
def resize_video(src_path, save_path, fps=24, target_h=480, target_w=720):
    cap = cv2.VideoCapture(src_path)
    assert(cap.isOpened(), f"fail to open video: {src_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (target_w, target_h))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (target_w, target_h))
        frames.append(resized_frame)
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"save video to {save_path}")

    return frames


parser = argparse.ArgumentParser()
parser.add_argument('src_path', type=str, default='./assets', help='Path to image/video to be edited')
parser.add_argument('device', type=str, default='cuda', help='cpu / cuda')
parser.add_argument('outdir', type=str, default='output/camera_ctrl', help='Path to output directory')
parser.add_argument('fps', type=int, default=24, help='Output video fps and total frames')
parser.add_argument('prompt', type=str, default=None, help='Prompt for generating the video')
args = parser.parse_args()

src_path = args.src_path
outdir = args.outdir
fps = args.fps
device = args.device
prompt = args.prompt

# if src is an image, turn to a video
src_name, src_ext = os.path.splitext(os.path.basename(src_path))
demo_dir = os.path.join(outdir, src_name)
os.path.exists(demo_dir) or os.makedirs(demo_dir)
img_ext = ['.jpg', '.jpeg', '.png']
video_path = f'{demo_dir}/{src_name}.mp4'

# prepare original video
if src_ext.lower() in img_ext:
    # image
    img_rgb = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    crop_img_path = f'{demo_dir}/{src_name}.png'
    first_frame_rgb = crop_rgb(img_rgb, save_path=crop_img_path)
    img2video(crop_img_path, video_path)
else:
    # video
    video_path = src_path
    first_frame_rgb = resize_video(src_path, video_path)[0]
# add to video.txt
vtxt_path = f'{demo_dir}/videos.txt'
with open(vtxt_path, 'a') as vtxt:
    vtxt.write(f'videos/{os.path.basename(video_path)}')

# prepare prompt
if not prompt:
    prompt = gen_prompt_for_rgb(first_frame_rgb)
if not prompt.endwith('\n'):
    prompt = prompt + '\n'
# write to prompt.txt
prompt_path = f'{demo_dir}/prompt.txt'
with open(prompt_path, 'a') as ptxt:
    ptxt.write(prompt)

# prepare tracking video
depth_model = Depth_pro(device=device)
