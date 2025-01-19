import PIL
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import cv2
from tqdm import tqdm
from PIL import Image

class Llava():
    def __init__(self,device='cuda:0',
                 llava_ckpt='llava-hf/bakLlava-v1-hf') -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def __call__(self,image:PIL.Image, prompt=None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
        
        prompt = '<image>\n USER: Detaily imagine and describe the scene this image taken from? \n ASSISTANT: This image is taken from a scene of ' if prompt is None else prompt
        inputs = self.processor(prompt, image, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer
    
    def _llava_prompt(self,frame):
        prompt = '<image>\n \
                USER: Detaily imagine and describe the scene this image taken from? \
                \n ASSISTANT: This image is taken from a scene of ' 
        return prompt 


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()
    return np.stack(frames)

def gen_prompt_for_rgb(rgb):
    model = Llava()
    query = '<image>\n \
            USER: Detaily imagine and describe the scene this image taken from? \
            \n ASSISTANT: This image is taken from a scene of '

    prompt = model(rgb, query)
    split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') + len(f'ASSISTANT: This image is taken from a scene of ')
    prompt = prompt[split:]
    print(f'prompt: {prompt}')
    return prompt
