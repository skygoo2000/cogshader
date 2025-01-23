import numpy as np

class Trajectory:
    def __init__(self, mode, num_frames=49):
        self.mode = mode
        self.num_frames = num_frames

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

    def spiral_camera_poses(self, num_frames, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
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