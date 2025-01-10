import cv2
import os

def extract_first_frame(video_path, output_path):
    video = cv2.VideoCapture(video_path)
    
    success, frame = video.read()
    
    if success:
        cv2.imwrite(output_path, frame)
        print(f"已保存 {output_path}")
    else:
        print(f"无法读取视频: {video_path}")
    
    video.release()

def process_directory(input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
            extract_first_frame(input_path, output_path)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(input_path, filename)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
                extract_first_frame(video_path, output_path)
    else:
        print(f"错误：{input_path} 既不是文件也不是目录")

if __name__ == "__main__":
    input_path = "/aifs4su/mmcode/lipeng/cogvideo/eval/blender/videos"
    output_directory = "/aifs4su/mmcode/lipeng/cogvideo/eval/blender/images"
    process_directory(input_path, output_directory)