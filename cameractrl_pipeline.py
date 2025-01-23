import os
import sys
import argparse
import cv2
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "submodules/MoGe"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip, ImageSequenceClip
from diffusers import FluxControlPipeline

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer
from submodules.MoGe.moge.model import MoGeModel
from image_gen_aux import DepthPreprocessor
from tests.inference import generate_video
from tools.gen_prompt import gen_prompt_for_rgb
from tools.trajectory import Trajectory

class CameractrlPipeline:
    def __init__(self, poses=None, gpu_id=0, output_dir='outputs', fov=55.0, frame_num=49, fps=24):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.max_depth = 65.0
        self.fov = fov
        self.fps = fps
        self.frame_num = frame_num
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)
        self.poses = poses.to(torch.float).to(self.device)
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
        
    def _load_video_frames(self, video_path, max_frames=49):
        """Load video frames"""
        video = VideoFileClip(video_path)
        frames = []
        for frame in video.iter_frames():
            frames.append(Image.fromarray(frame))
            
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        elif len(frames) < max_frames:
            last_frame = frames[-1]
            while len(frames) < max_frames:
                frames.append(last_frame.copy())
        
        return torch.stack([self.transform(frame) for frame in frames])
    
    def _process_image(self, image_path, max_frames=49):
        """Process single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).repeat(max_frames, 1, 1, 1)
    
    def _tensor2video(self, video_tensor, output_path, fps=24, W=720, H=480):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        video_tensor = video_tensor.permute(0, 2, 3, 1).cpu().numpy() # [T, H, W, 3]
        for frame in video_tensor:
            video_writer.write(frame)
        video_writer.release()

    def _get_intr(self, fov, H=480, W=720):
        fov_rad = math.radians(fov)
        # 计算焦距 (focal length)，基于宽度的水平 FOV
        focal_length = (W / 2) / math.tan(fov_rad / 2)

        # 假设光心在图像中心
        cx = W / 2
        cy = H / 2

        # 构造内参矩阵
        intr = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32)

        return intr

    def _apply_poses(self, pts, intr, poses):
        T, N, _ = pts.shape
        ones = torch.ones(T, N, 1, device=self.device, dtype=torch.float)
        pts_hom = torch.cat([pts[:, :, :2], ones], dim=-1)  # (T, N, 3)
        pts_cam = torch.bmm(pts_hom, torch.linalg.inv(intr).transpose(1, 2))  # (T, 3, N)
        pts_cam[:,:, :3] *= pts[:, :, 2:3]

        pts_cam = torch.cat([pts_cam, ones], dim=-1)  # (T, N, 4)
        pts_world = torch.bmm(pts_cam, poses.transpose(1, 2))[:, :, :3] # (T, N, 3)

        pts_proj = torch.bmm(pts_world, intr.transpose(1, 2))  # (T, 3, N)
        pts_proj[:, :, :2] /= pts_proj[:, :, 2:3]

        return pts_proj

    def generate_tracking(self, video_tensor):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            
        Returns:
            str: Path to tracking video
        """
        print("Loading tracking models...")
        # Load tracking model
        tracker = SpaTrackerPredictor(
            checkpoint=os.path.join(project_root, 'checkpoints/spaT_final.pth'),
            interp_shape=(384, 576),
            seq_length=12
        ).to(self.device)
        
        # Load depth model
        self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.depth_preprocessor.to(self.device)
        
        try:
            video = video_tensor.unsqueeze(0).to(self.device)
            
            video_depths = []
            for i in range(video_tensor.shape[0]):
                frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                depth = self.depth_preprocessor(Image.fromarray(frame))[0]
                depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
                video_depths.append(depth_tensor)
            video_depth = torch.stack(video_depths, dim=0).to(self.device)
            # print("Video depth shape:", video_depth.shape)
            
            segm_mask = np.ones((480, 720), dtype=np.uint8)
            
            pred_tracks, pred_visibility, T_Firsts = tracker(
                video, 
                video_depth=video_depth,
                grid_size=70,
                backward_tracking=False,
                depth_predictor=None,
                grid_query_frame=0,
                segm_mask=torch.from_numpy(segm_mask)[None, None].to(self.device),
                wind_length=12,
                progressive_tracking=False
            )

            return pred_tracks, pred_visibility, T_Firsts
            
        finally:
            # Clean up GPU memory
            del tracker, self.depth_preprocessor
            torch.cuda.empty_cache()
    
    def apply_traj_on_tracking(self, pred_tracks, poses, fov=55, frame_num=49):
        intr = self._get_intr(fov).unsqueeze(0).repeat(frame_num, 1, 1).to(self.device)
        tracking_pts = self._apply_poses(pred_tracks.squeeze(), intr, poses).unsqueeze(0)
        return tracking_pts

    def visualize_tracking(self, video, pred_tracks, pred_visibility, T_Firsts, save_tracking=True):
        video = video.unsqueeze(0).to(self.device)
        vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
        msk_query = (T_Firsts == 0)
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(video=video, tracks=pred_tracks,
                        visibility=pred_visibility, save_video=False, filename='temp')
        
        tracking_video = tracking_video.squeeze(0) # [T, C, H, W]
        wide_list = list(tracking_video.unbind(0))
        wide_list = [wide.permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        clip = ImageSequenceClip(wide_list, fps=self.fps)

        tracking_path = None
        if save_tracking:
            try:
                tracking_path = os.path.join(self.output_dir, "tracking_video.mp4")
                clip.write_videofile(tracking_path, codec="libx264", fps=self.fps, logger=None)
                print(f"Video saved to {tracking_path}")
            except Exception as e:
                print(f"Warning: Failed to save tracking video: {e}")
                tracking_path = None
        
        return tracking_path, tracking_video
    
    # def generate_tracking(self, video_tensor, fov=55):
    #     """Generate tracking video
        
    #     Args:
    #         video_tensor (torch.Tensor): Input video tensor
    #         trajs (torch.Tensor): camera poses for each frame
    #         fov (int): vertical fov of the camera
    #     Returns:
    #         str: Path to tracking video
    #     """
    #     print("Loading tracking models...")
    #     # Load tracking model
    #     tracker = SpaTrackerPredictor(
    #         checkpoint=os.path.join(project_root, 'checkpoints/spaT_final.pth'),
    #         interp_shape=(384, 576),
    #         seq_length=12
    #     ).to(self.device)
        
    #     # Load depth model
    #     self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    #     self.depth_preprocessor.to(self.device)
        
    #     try:
    #         video = video_tensor.unsqueeze(0).to(self.device)
            
    #         video_depths = []
    #         for i in range(video_tensor.shape[0]):
    #             frame = (video_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #             depth = self.depth_preprocessor(Image.fromarray(frame))[0]
    #             depth_tensor = transforms.ToTensor()(depth)  # [1, H, W]
    #             video_depths.append(depth_tensor)
    #         video_depth = torch.stack(video_depths, dim=0).to(self.device) # [T, 1, H, W]
    #         print("Video depth shape:", video_depth.shape)
            
    #         segm_mask = np.ones((480, 720), dtype=np.uint8)
            
    #         pred_tracks, pred_visibility, T_Firsts = tracker(
    #             video, 
    #             video_depth=video_depth,
    #             grid_size=70,
    #             backward_tracking=False,
    #             depth_predictor=None,
    #             grid_query_frame=0,
    #             segm_mask=torch.from_numpy(segm_mask)[None, None].to(self.device),
    #             wind_length=12,
    #             progressive_tracking=False
    #         )
            
    #         # apply pose
    #         import ipdb;ipdb.set_trace()

    #         intr = self._get_intr(self.fov).unsqueeze(0).repeat(self.frame_num, 1, 1).to(self.device)
    #         poses = torch.from_numpy(self._spiral_camera_poses(self.frame_num, 1)).to(torch.float).to(self.device)
    #         pts = self._apply_poses(pred_tracks.squeeze(), intr, poses).unsqueeze(0)
            

    #         vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
    #         msk_query = (T_Firsts == 0)
    #         pred_tracks = pts[:,:,msk_query.squeeze()]
    #         pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
            
    #         tracking_path = os.path.join(self.output_dir, "tracking/temp_tracking.mp4")



    #         vis.visualize(video=video, tracks=pred_tracks,
    #                      visibility=pred_visibility,
    #                      filename="temp")
            
    #         return tracking_path
            
    #     finally:
    #         # Clean up GPU memory
    #         del tracker, self.depth_preprocessor
    #         torch.cuda.empty_cache()
    
    def apply_tracking(self, input_path=None, tracking_path=None, repainted_path=None, prompt=None, checkpoint_path=None):
        """Generate final video with motion transfer
        
        Args:
            input_path (str): Path to input video/image
            tracking_path (str): Path to tracking video (optional)
            repainted_path (str): Path to repainted image (optional)
            prompt (str): Generation prompt
            checkpoint_path (str): Path to model checkpoint
        """
        print("Processing input...")
        # Load input
        input_ext = os.path.splitext(input_path)[1].lower()
        if input_ext in ['.mp4', '.avi', '.mov']:
            video_tensor = self._load_video_frames(input_path)
        else:
            video_tensor = self._process_image(input_path)
        if not prompt:
            prompt = gen_prompt_for_rgb(video_tensor[0].detach().cpu().numpy())
        # Generate tracking
        pred_tracks, pred_visibility, T_Firsts = self.generate_tracking(video_tensor) if tracking_path is None else tracking_path
        # Apply trajectory
        if not self.poses is None:
            pred_tracks = self.apply_traj_on_tracking(pred_tracks, self.poses)
        # Visualize
        # import ipdb;ipdb.set_trace()
        filename = 'temp'
        tracking_path, _ = self.visualize_tracking(video_tensor, pred_tracks, pred_visibility, T_Firsts, save_tracking=True)
        print(f"Tracking video generated at: {tracking_path}")
        
        # Generate final video
        final_output = os.path.join(os.path.abspath(self.output_dir), f"cameractrl_from[{os.path.splitext(os.path.basename(input_path))[0]}].mp4")
        generate_video(
            prompt=prompt,
            model_path=checkpoint_path,
            tracking_path=os.path.abspath(tracking_path),
            image_or_video_path=os.path.abspath(input_path),
            output_path=final_output,
            num_inference_steps=50,
            guidance_scale=6.0,
            generate_type="i2v",
            dtype=torch.bfloat16
        )
        print(f"Final video generated successfully at: {final_output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, default=None, help='prompt for the input')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--fps', type=int, default=8, help='fps of the generated video')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float)

    Traj = Trajectory("spiral")
    poses = Traj.spiral_camera_poses(49, 1)
    Cameractrl = CameractrlPipeline(gpu_id=args.gpu, output_dir=args.output_dir, poses=poses, fps=args.fps)
    
    # motion transfer
    Cameractrl.apply_tracking(input_path=args.input_path, prompt=args.prompt, checkpoint_path=args.checkpoint_path)

if __name__ == "__main__":
    main() 