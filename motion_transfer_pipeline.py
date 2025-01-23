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
from moviepy.editor import ImageSequenceClip
class MotionTransferPipeline:
    def __init__(self, gpu_id=0, output_dir='outputs'):
        """Initialize MotionTransfer class
        
        Args:
            gpu_id (int): GPU device ID
            output_dir (str): Output directory path
        """
        self.max_depth = 65.0
        self.device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        self.output_dir = output_dir
        self.fps = 8
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.Resize((480, 720)),
            transforms.ToTensor()
        ])
        
    def _load_video_frames(self, video_path, max_frames=49):
        """Load video frames"""
        video = VideoFileClip(video_path)
        self.fps = video.fps
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

    def visualize_tracking(self, video, pred_tracks, pred_visibility, T_Firsts, save_tracking=False):
        video = video.unsqueeze(0).to(self.device)
        vis = Visualizer(save_dir=self.output_dir, grayscale=False, fps=24, pad_value=0)
        msk_query = (T_Firsts == 0)
        pred_tracks = pred_tracks[:,:,msk_query.squeeze()]
        pred_visibility = pred_visibility[:,:,msk_query.squeeze()]
        
        tracking_video = vis.visualize(video=video, tracks=pred_tracks,
                        visibility=pred_visibility, save_video=False,
                        filename="temp")
        
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
    
    def repaint_first_frame(self, video_tensor, prompt, depth_path=None, method="dav"):
        """Repaint first frame using Flux
        
        Args:
            video_tensor (torch.Tensor): Input video tensor
            prompt (str): Repaint prompt
            depth_path (str): Path to depth image
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
        if depth_path is None:
            if method == "moge":
                self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
                first_frame_tensor = video_tensor[0].to(self.device)
                depth_map = self.moge_model.infer(first_frame_tensor)["depth"]
                depth_map = torch.clamp(depth_map, max=self.max_depth)
                depth_normalized = 1.0 - (depth_map / self.max_depth)
                depth_rgb = (depth_normalized * 255).cpu().numpy().astype(np.uint8)
                depth_rgb_path = os.path.join(self.output_dir, "depth_moge.png")
                Image.fromarray(depth_rgb).save(depth_rgb_path)
                control_image = Image.fromarray(depth_rgb).convert("RGB")
            elif method == "zoedepth":
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("Intel/zoedepth-nyu-kitti")
                first_frame = (video_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(first_frame))[0].convert("RGB")
                control_image = control_image.point(lambda x: 255 - x) # the zoedepth depth is inverted
                control_image.save(os.path.join(self.output_dir, "depth_zoe.png"))
            else:
                self.depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                first_frame = (video_tensor[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_image = self.depth_preprocessor(Image.fromarray(first_frame))[0].convert("RGB")
                control_image.save(os.path.join(self.output_dir, "depth_dav.png"))
        else:
            self.depth_preprocessor = None
            control_image = Image.open(depth_path).convert("RGB")

        try:

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
    
    def apply_tracking(self, input_path=None, tracking_path=None, repainted_path=None, prompt=None, checkpoint_path=None, depth_path=None):
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
        
        if tracking_path is None:
            # Generate tracking video
            pred_tracks, pred_visibility, T_Firsts = self.generate_tracking(video_tensor)
            tracking_path, tracking_video = self.visualize_tracking(video_tensor, pred_tracks, pred_visibility, T_Firsts)
        else:
            tracking_video = None
            print(f"Tracking video loaded from {tracking_path}")
        
        if repainted_path is None:
            # Repaint first frame
            repainted_path = self.repaint_first_frame(video_tensor, prompt, depth_path=depth_path)
            repainted_path = os.path.abspath(repainted_path)
            print(f"First frame repainted at: {repainted_path}")
        else:
            print(f"Repainted image loaded from {repainted_path}")
        
        # Generate final video
        final_output = os.path.join(os.path.abspath(self.output_dir), f"motion_transfer_from[{os.path.splitext(os.path.basename(input_path))[0]}].mp4")
        generate_video(
            prompt=prompt,
            model_path=checkpoint_path,
            tracking_path=tracking_path, # tracking_path has higher priority
            tracking_video=tracking_video,
            image_or_video_path=repainted_path,
            output_path=final_output,
            num_inference_steps=50,
            guidance_scale=6.0,
            generate_type="i2v",
            dtype=torch.bfloat16,
            fps=self.fps
        )
        print(f"Final video generated successfully at: {final_output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None, help='Path to input video/image')
    parser.add_argument('--prompt', type=str, required=True, help='Repaint prompt')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--depth_path', type=str, default=None, help='Path to depth image')
    parser.add_argument('--tracking_path', type=str, default=None, help='Path to tracking video')
    parser.add_argument('--repaint_path', type=str, default=None, help='Path to repainted image')
    args = parser.parse_args()
    
    motion_transfer = MotionTransferPipeline(gpu_id=args.gpu, output_dir=args.output_dir)
    
    # motion transfer
    # motion_transfer.apply_tracking(input_path=args.input_path, prompt=args.prompt, checkpoint_path=args.checkpoint_path)

    # animating mesh to video
    motion_transfer.apply_tracking(input_path=args.input_path, prompt=args.prompt, checkpoint_path=args.checkpoint_path, depth_path=args.depth_path, tracking_path=args.tracking_path, repainted_path=args.repaint_path)

if __name__ == "__main__":
    main() 