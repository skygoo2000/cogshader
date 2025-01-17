import json
import os

def read_file_lines(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def generate_metadata(base_path):
    parent_folder = os.path.basename(base_path)
    videos = read_file_lines(f"{base_path}/videos.txt")
    trackings = read_file_lines(f"{base_path}/trackings.txt")
    prompts = read_file_lines(f"{base_path}/prompt.txt")
    
    metadata = []
    min_length = min(len(videos), len(trackings), len(prompts))
    
    for i in range(min_length):
        entry = {
            "video": f"{parent_folder}/{videos[i]}",
            "tracking_video": f"{parent_folder}/{trackings[i]}",
            "text": prompts[i],
            "aes_score": 6.5
        }
        metadata.append(entry)
    
    output_path = f"{base_path}/{parent_folder}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"generted dataset: {parent_folder} metadata, include {len(metadata)} samples")

if __name__ == "__main__":
    base_path = "/aifs4su/mmcode/lipeng/cogvideo/avatars/trackingavatars"
    generate_metadata(base_path) 