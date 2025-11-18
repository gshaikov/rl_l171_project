# Author: Hantao Zhong
# Date: August 2025
# IK = Inverse Kinematics: find the joint angles to reach a desired end-effector pose

import mujoco
import numpy as np
import time
from pathlib import Path
try:
    import viser
    from viser.extras import ViserUrdf
    import yourdfpy
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False

DAMPING_COEFF = 1e-12
MAX_ANGLE_CHANGE = np.deg2rad(45)

class IKSolver:
    def __init__(self, enable_viz=False):
        # Load arm without gripper
        self.model = mujoco.MjModel.from_xml_path('rl_l171/assets/xmls/piper/piper.xml')
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        self.qpos0 = self.model.key('home').qpos
        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)
        self.eye = np.eye(self.model.nv)

        # Initialize visualization if requested
        self.enable_viz = enable_viz and VISER_AVAILABLE
        self.server = None
        self.viser_urdf = None
        self.target_frame = None
        self.result_frame = None
        
        if self.enable_viz:
            self._setup_visualization()

    def _setup_visualization(self):
        """Initialize viser server and robot visualization"""
        # Start viser server
        self.server = viser.ViserServer()
        
        # Load URDF for visualization
        urdf_path = "models/piper_description.urdf"
        urdf_file = Path(urdf_path)
        if not urdf_file.exists():
            print(f"Warning: URDF file not found for visualization: {urdf_path}")
            self.enable_viz = False
            return
    
        try:
            urdf = yourdfpy.URDF.load(
                str(urdf_file),
                load_meshes=True,
                build_scene_graph=True,
                load_collision_meshes=False,
                build_collision_scene_graph=False,
            )
            
            self.viser_urdf = ViserUrdf(
                self.server,
                urdf_or_path=urdf,
                load_meshes=True,
                load_collision_meshes=False,
            )
            
            # Create coordinate frames for target and result
            self.target_frame = self.server.scene.add_frame(
                "/target_pose",
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.1,
                axes_radius=0.005,
            )
            
            self.result_frame = self.server.scene.add_frame(
                "/result_pose", 
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                axes_length=0.08,
                axes_radius=0.003,
            )
            
            # Add grid
            self.server.scene.add_grid(
                "/grid",
                width=1,
                height=1,
                position=(0.0, 0.0, 0.0),
            )
            
            print("Viser visualization initialized. Check http://localhost:8080")
            
        except Exception as e:
            print(f"Warning: Failed to setup visualization: {e}")
            self.enable_viz = False

    def solve(self, pos, quat, curr_qpos, max_iters=20, err_thresh=1e-4):
        # Convert quaternion from (x, y, z, w) to (w, x, y, z) for mujoco
        quat_mj = quat[[3, 0, 1, 2]]
        
        # Update target frame visualization
        if self.enable_viz and self.target_frame is not None:
            # Convert quaternion to (w, x, y, z) for viser
            quat_viser = quat[[3, 0, 1, 2]]  # (x,y,z,w) -> (w,x,y,z)
            self.target_frame.wxyz = tuple(quat_viser)
            self.target_frame.position = tuple(pos)

        # Set arm to initial joint configuration
        # Ensure curr_qpos is length 8 (pad with zeros if needed)
        if len(curr_qpos) < 8:
            curr_qpos = np.concatenate([curr_qpos, np.zeros(8 - len(curr_qpos))])
        self.data.qpos = curr_qpos

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)

            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat_mj, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            # Calculate update
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)
            # Add secondary task to be as close as possible to home position, defined in the keyframe "home" in the xml
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (self.eye - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > MAX_ANGLE_CHANGE:
                update *= MAX_ANGLE_CHANGE / update_max

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)
            
            # Update robot visualization during IK solving
            if self.enable_viz and self.viser_urdf is not None:
                self.viser_urdf.update_cfg(self.data.qpos[:self.model.nv])

        # Update result frame with final end-effector pose
        if self.enable_viz and self.result_frame is not None:
            # Recompute kinematics for final result
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            
            # Get final end-effector pose
            final_pos = self.site_pos.copy()
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            # Convert from (w,x,y,z) to (w,x,y,z) for viser (already in correct format)
            final_quat_viser = self.site_quat.copy()
            
            self.result_frame.position = tuple(final_pos)
            self.result_frame.wxyz = tuple(final_quat_viser)

        return self.data.qpos.copy()