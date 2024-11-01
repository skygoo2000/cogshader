import io,os

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

MODEL_PATH = "THUDM/cogvlm2-llama3-caption"
DATASET_ROOT = '/aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira200'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
args = parser.parse_args([])


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def test():
    prompt = "Please describe this video in detail."
    temperature = 0.1
    video_data = open('test.mp4', 'rb').read()
    response = predict(prompt, video_data, temperature)
    print(response)

def caption_dataset(video_list_file, output_file):
    prompt = "Please describe this video in detail."
    temperature = 0.1
    
    # 获取视频列表文件的目录路径
    base_dir = os.path.dirname(os.path.abspath(video_list_file))
    
    with open(video_list_file, 'r') as f:
        video_paths = f.read().splitlines()
    
    with open(output_file, 'w') as f:
        for video_path in video_paths:
            if not video_path:  # 跳过空行
                continue
            
            # 如果是相对路径，转换为绝对路径
            if not os.path.isabs(video_path):
                video_path = os.path.join(base_dir, video_path)
                
            print(f"正在处理视频: {video_path}")
            try:
                video_data = open(video_path, 'rb').read()
                response = predict(prompt, video_data, temperature)
                f.write(f"{video_path}\t{response}\n")
                f.flush()  # 确保实时写入
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {str(e)}")

if __name__ == '__main__':
    video_list_file = os.path.join(DATASET_ROOT, 'videos.txt')
    output_file = os.path.join(DATASET_ROOT, 'prompts.txt')
    caption_dataset(video_list_file, output_file)
