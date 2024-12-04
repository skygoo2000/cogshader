import os
import shutil

def merge_datasets(source_dir, target_dir):
    """
    合并两个数据集目录
    source_dir: 源数据集目录
    target_dir: 目标数据集目录
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 复制 tracking 文件夹中的文件
    source_tracking = os.path.join(source_dir, 'tracking')
    target_tracking = os.path.join(target_dir, 'tracking')
    os.makedirs(target_tracking, exist_ok=True)
    
    for item in os.listdir(source_tracking):
        source_item = os.path.join(source_tracking, item)
        target_item = os.path.join(target_tracking, item)
        if os.path.isfile(source_item):
            shutil.copy2(source_item, target_item)
            
    # 复制 videos 文件夹中的文件
    source_videos = os.path.join(source_dir, 'videos')
    target_videos = os.path.join(target_dir, 'videos')
    os.makedirs(target_videos, exist_ok=True)
    
    for item in os.listdir(source_videos):
        source_item = os.path.join(source_videos, item)
        target_item = os.path.join(target_videos, item)
        if os.path.isfile(source_item):
            shutil.copy2(source_item, target_item)
    
    # 合并文本文件
    text_files = ['prompt.txt', 'videos.txt', 'trackings.txt']
    for file_name in text_files:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        
        # 如果源文件存在
        if os.path.exists(source_file):
            # 如果目标文件不存在，直接复制
            if not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
            else:
                # 如果目标文件存在，追加内容
                # 首先读取目标文件内容并删除末尾空行
                with open(target_file, 'r', encoding='utf-8') as target:
                    target_content = target.read().rstrip()
                
                # 读取源文件内容
                with open(source_file, 'r', encoding='utf-8') as source:
                    source_content = source.read().strip()
                
                # 写入处理后的内容
                with open(target_file, 'w', encoding='utf-8') as target:
                    if target_content:
                        target.write(target_content + '\n' + source_content)
                    else:
                        target.write(source_content)

if __name__ == "__main__":
    # 设置源数据集和目标数据集的路径
    source_dataset = "/aifs4su/mmcode/lipeng/cogvideo/datasets/trackingavatars"
    target_dataset = "/aifs4su/mmcode/lipeng/cogvideo/datasets/cogmira200"
    
    # 执行合并操作
    merge_datasets(source_dataset, target_dataset)
    print("数据集合并完成！") 