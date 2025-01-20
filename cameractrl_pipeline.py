import os
import sys
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "submodules/MoGe"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from moviepy.editor import VideoFileClip
from diffusers import FluxControlPipeline

from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer
from submodules.MoGe.moge.model import MoGeModel
from image_gen_aux import DepthPreprocessor
from tests.inference import generate_video


class CameractrlPipeline:
    def __init__(self, gpu_id=0, output_dir='outputs', fov=55.0, frame_num=49):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.max_depth = 65.0
        self.fov = fov
        self.frame_num = frame_num
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
    
    def _apply_pose(pts, intr, extr):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Nx4
        # Apply the transformation
        pts_hom = pts_hom @ extr.T  # Nx4

        # Normalize to 3D coordinates
        pts = pts_hom[:, :3] / pts_hom[:, 3]
        depths = pts[:, 2]
        # Project back to 2D image plane using intrinsic matrix
        pts_hom = pts @ intr.T
        pts = pts_hom[:, :2] / pts_hom[:, 2]

        return pts, depths

    def _spiral_trajectory(self, num_frames, radius, forward_ratio=0.2, backward_ratio=0.8):
        t = np.linspace(0, 1, num_frames)  # 保持 t 从 0 到 1
        r = np.sin(np.pi * t) * radius
        theta = 2 * np.pi * t  
        
        # not to change y much (up-down for floor and sky)
        y = r * np.cos(theta) * 0.3
        x = r * np.sin(theta)
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
    
        return x, y, z

    def _look_at(self, camera_position, target_position):
        # look at direction
        # import ipdb;ipdb.set_trace()
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def _spiral_camera_poses(self, num_frames, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        x, y, z = self._spiral_trajectory(num_frames, radius * rotation_times, forward_ratio, backward_ratio)
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3,  3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)

        return camera_poses
    
    def _get_intr(self, fov, H=480, W=720):
        f = H / (2 * torch.tan(fov / 2))
        intr = torch.zeros((3, 3))
        intr[0, 0] = f
        intr[1, 1] = f
        intr[0, 2] = W / 2
        intr[1, 2] = H / 2
        return intr

    def generate_tracking(self, video_tensor, trajs, fov):
        """Generate tracking video
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            trajs (torch.Tensor): camera poses for each frame
            fov (int): vertical fov of the camera
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
            print("Video depth shape:", video_depth.shape)
            
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
            
            # apply pose
            intr = self._get_intr(self.fov).repeat(self.frame_num, 1)
            # 1. 将 (u, v) 转换为齐次坐标 (u, v, 1)
            ones = torch.ones(pred_tracks.shape[:-1] + [1], device=self.device)  # (N, H, W, 1)
            pts_hom = torch.cat([pred_tracks, ones], dim=-1)  # (N, H, W, 3)

            # (N, 3, 3) @ (N, H, W, 3) -> (N, H, W, 3)
            points_camera = torch.einsum('nij,nijh->nihw', torch.linalg.inv(intrinsic), points_homogeneous)

            # 4. 乘以深度，将归一化坐标转换到相机坐标
            points_camera = points_camera * depth.unsqueeze(-1)  # (N, H, W, 3)

            # 5. 将相机坐标扩展为齐次坐标 (X, Y, Z, 1)
            ones = torch.ones((N, H, W, 1), device=points.device)  # (N, H, W, 1)
            points_camera_homogeneous = torch.cat([points_camera, ones], dim=-1)  # (N, H, W, 4)

            # 6. 使用位姿矩阵转换到世界坐标
            # (N, 4, 4) @ (N, H, W, 4) -> (N, H, W, 4)
            points_world = torch.einsum('nij,nijh->nihw', pose, points_camera_homogeneous)

            # 7. 去掉齐次坐标最后一维
            points_world = points_world[..., :3]  # (N, H, W, 3)


            vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
            msk_query = (T_Firsts == 0)
            pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
            pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
            
            tracking_path = os.path.join(self.output_dir, "tracking/temp_tracking.mp4")



            vis.visualize(video=video, tracks=pred_tracks,
                         visibility=pred_visibility,
                         filename="temp")
            
            return tracking_path
            
        finally:
            # Clean up GPU memory
            del tracker, self.depth_preprocessor
            torch.cuda.empty_cache()
    
    def repaint_first_frame(self, video_tensor, prompt, method="dav"):
        """Repaint first frame using Flux
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            prompt (str): Repaint prompt
            method (str): depth estimator, "moge" or "dav"
        Returns:
            str: Path to repainted image
        """
        print("Loading Flux model...")
        # Load Flux model
        flux_pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev", 
            torch_dtype=torch.bfloat16
        ).to(self.device)

        # Load model
        if method == "moge":
            self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
        elif method == "zoedepth":
            self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        else:
            self.depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

        try:
            if method == "moge":
                first_frame_tensor = video_tensor[0].to(self.device)
                depth_map = self.moge_model.infer(first_frame_tensor)["depth"]
                depth_map = torch.clamp(depth_map, max=self.max_depth)
                depth_normalized = 1.0 - (depth_map / self.max_depth)
                depth_rgb = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
                depth_rgb_path = os.path.join(self.output_dir, "depth_moge.png")
                Image.fromarray(depth_rgb).save(depth_rgb_path)
                control_image = Image.fromarray(depth_rgb).convert("RGB")
            elif method == "zoedepth":
                first_frame = (video_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(first_frame))[0].convert("RGB")
                control_image = control_image.point(lambda x: 255 - x) # the zoedepth depth is inverted
                control_image.save(os.path.join(self.output_dir, "depth_zoe.png"))
            else:
                first_frame = (video_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(first_frame))[0].convert("RGB")
                control_image.save(os.path.join(self.output_dir, "depth_dav.png"))
            
            repainted_image = flux_pipe(
                prompt=prompt,
                control_image=control_image,
                height=480,
                width=720,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
            
            repainted_path = os.path.join(self.output_dir, "repainted.png")
            repainted_image.save(repainted_path)
            return repainted_path
            
        finally:
            # Clean up GPU memory
            del flux_pipe
            if method == "moge":
                del self.moge_model
            else:
                del self.depth_preprocessor
            torch.cuda.empty_cache()
    
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
            
        # Generate tracking video
        tracking_path = self.generate_tracking(video_tensor) if tracking_path is None else tracking_path
        print(f"Tracking video generated at: {tracking_path}")
        
        # Repaint first frame
        repainted_path = self.repaint_first_frame(video_tensor, prompt) if repainted_path is None else repainted_path
        print(f"First frame repainted at: {repainted_path}")
        
        # Generate final video
        final_output = os.path.join(os.path.abspath(self.output_dir), f"motion_transfer_from[{os.path.splitext(os.path.basename(input_path))[0]}].mp4")
        generate_video(
            prompt=prompt,
            model_path=checkpoint_path,
            tracking_path=os.path.abspath(tracking_path),
            image_or_video_path=os.path.abspath(repainted_path),
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
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    motion_transfer = CameractrlPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    
    # motion transfer
    motion_transfer.apply_tracking(input_path=args.input_path, prompt=args.prompt, checkpoint_path=args.checkpoint_path)

if __name__ == "__main__":
    main() 