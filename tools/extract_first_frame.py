import cv2
import os

def extract_first_frame(video_path, output_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 读取第一帧
    success, frame = video.read()
    
    if success:
        # 保存第一帧为JPG
        cv2.imwrite(output_path, frame)
        print(f"已保存 {output_path}")
    else:
        print(f"无法读取视频: {video_path}")
    
    # 释放视频对象
    video.release()

def process_directory(input_path, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isfile(input_path):
        # 如果输入是单个文件
        filename = os.path.basename(input_path)
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            extract_first_frame(input_path, output_path)
    elif os.path.isdir(input_path):
        # 如果输入是目录
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(input_path, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
                extract_first_frame(video_path, output_path)
    else:
        print(f"错误：{input_path} 既不是文件也不是目录")

if __name__ == "__main__":

    input_path = "/aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira/videos/000000048_0.mp4"
    output_directory = "/aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira200"
    process_directory(input_path, output_directory)