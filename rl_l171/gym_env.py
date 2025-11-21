import time
import itertools

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from rl_l171.utils.constants import POLICY_CONTROL_PERIOD, INSIDE_THRESHOLD
from rl_l171.mujoco_env import CubesMujocoSim


class CubesGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        max_nr_steps=100,
        randomise_initial_position=False,
        seed=None,
        nr_cubes=5,
    ):
        super().__init__()

        # The underlying environment that handles the MuJoCo simulation
        self.sim = CubesMujocoSim(
            randomise_initial_position=randomise_initial_position,
            seed=seed,
            nr_cubes=nr_cubes,
        )
        self.render_mode = render_mode

        # Viewer handle if needed
        self.viewer = None
        self.renderer = None

        if render_mode == "rgb_array":
            # Initialize the offscreen renderer
            from mujoco.renderer import Renderer

            self.renderer = Renderer(self.sim.model)
        elif render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(
                self.sim.model,
                self.sim.data,
                show_left_ui=False,
                show_right_ui=False,
            )

        # Number of mujoco simulation steps to execute per RL action
        self.steps_per_action = int(POLICY_CONTROL_PERIOD / self.sim.model.opt.timestep)

        # assert nr_cubes in [5, 10], "You should only focus  on 5 or 10 cubes!"
        self.nr_cubes = nr_cubes

        self._max_episode_steps = max_nr_steps
        self.current_step = 0

        # Action: [base_delta_x, base_delta_y, base_delta_THETA,
        #          arm_delta_x, arm_delta_y, arm_delta_z,
        #          gripper]
        low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0])
        high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # vals = [-0.1, 0.0, 0.1]
        #
        # self.ACTION_TABLE = np.array([
        #     [bx, by, btheta, 0.0, 0.0, 0.0, 0.0]
        #     for bx, by, btheta in itertools.product(vals, repeat=3)
        # ], dtype=np.float32)
        #
        # Now the agent chooses an *index* into this table
        # self.action_space = spaces.Discrete(self.ACTION_TABLE.shape[0])  # 27 actions

        # Observation space
        observation_space_dict = {
            # Robot state: [base_x, base_y, base_theta, arm_x, arm_y, arm_z, arm_quat_w, arm_quat_x, arm_quat_y, arm_quat_z, gripper_pos x2]
            "robot_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            ),
        }
        for j in range(nr_cubes):
            observation_space_dict[f"cube_{j}_pos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_space_dict)

    def _calculate_reward(self, obs: dict) -> tuple[float, bool]:
        """Calculates reward and determines if the episode is done."""

        # (N, 2) array of current cube 2D positions
        current_pos_array = np.array(
            [obs[f"cube_{j}_pos"][:2] for j in range(self.nr_cubes)]
        )

        # (N, 2) array of target positions -- the target is the origin!
        cube_distances = np.linalg.norm(current_pos_array, axis=1)

        # reward is the average of the negative distances of the cubes and the robot to the target
        # if distance less than INSIDE_THRESHOLD, zero reward
        cube_distances[cube_distances <= INSIDE_THRESHOLD] = 0
        reward = -np.mean(cube_distances)

        completed = np.all(cube_distances < INSIDE_THRESHOLD)
        if completed:
            # we do not terminate the episode on purpose as we want the agent to learn to keep the cubes inside the target area
            print("All cubes collected!")

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)

        self.current_step = 0

        _obs = self._agent_obs()
        _info = {}
        return _obs, _info

    def _full_observation(self):
        """Get the full observation from the simulation state."""
        state = self.sim.get_state()

        # Enforce quaternion uniqueness
        arm_quat = state["arm_quat_global_wxyz"].copy()
        if arm_quat[0] < 0.0:
            np.negative(arm_quat, out=arm_quat)

        _obs = {
            "base_pose": state["base_pose"],
            "arm_pos_global": state["arm_pos_global"],
            "arm_quat_global_wxyz": arm_quat,
            "gripper_pos": state["gripper_pos"],
        }

        # Add cube positions
        for i in range(self.nr_cubes):
            _obs[f"cube_{i}_pos"] = state[f"cube_{i}_pos"]
        return _obs

    def _agent_obs(self) -> dict:
        """Get the observation we pass the agent."""
        full_obs = self._full_observation()

        robot_state = np.hstack(
            [
                full_obs["base_pose"],
                full_obs["arm_pos_global"],
                full_obs["arm_quat_global_wxyz"],
                full_obs["gripper_pos"],
            ]
        )

        _obs = {"robot_state": robot_state}

        for j in range(self.nr_cubes):
            _obs[f"cube_{j}_pos"] = full_obs[f"cube_{j}_pos"]

        return _obs

    def step(self, action: np.ndarray):
        # action is now a discrete index (int or np.int64)
        # action_idx = int(action)
        # action_vec = self.ACTION_TABLE[action_idx]  # shape (7,)

        # Get current end effector pose
        current_obs = self._full_observation()
        current_base_pose = current_obs["base_pose"]
        current_arm_pos = current_obs["arm_pos_global"]
        current_quat = current_obs["arm_quat_global_wxyz"]  # in [w, x, y, z]
        target_quat_xyzw = current_quat[[1, 2, 3, 0]]

        # Apply the delta action to get the target pose
        base_action = action[:3]
        arm_action = action[3:6]
        gripper_action = action[-1]

        target_base_pos = current_base_pose + base_action
        target_arm_pos = current_arm_pos + arm_action

        # Create command with both base and arm targets
        command = {
            "base_pose": target_base_pos,
            "arm_pos_global": target_arm_pos,
            "arm_quat_global_xyzw": target_quat_xyzw,
            "gripper_pos": np.array([gripper_action]),
        }

        # Execute multiple mujoco simulation steps for this action
        for step_idx in range(self.steps_per_action):
            if step_idx == 0:
                self.sim.step_simulation(command)
            else:
                self.sim.step_simulation(None)  # continue with last command

            if self.viewer is not None:
                self.viewer.sync()

        # Get new observation, calculate reward, and check termination
        _obs = self._agent_obs()
        _reward = self._calculate_reward(_obs)

        self.current_step += 1
        _truncated = self.current_step >= self._max_episode_steps

        _info = {}

        return _obs, _reward, False, _truncated, _info

    def render(self):
        if self.renderer:
            # Update the scene with the latest simulation data
            self.renderer.update_scene(self.sim.data, camera="overview_cam")
            # Render the scene and return the pixel data
            return self.renderer.render()
        # If not in rgb_array mode, this can return None
        return None

    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    args = parser.parse_args()

    render_mode = None
    if args.render:
        render_mode = "human"

    env = CubesGymEnv(
        render_mode=render_mode,
        max_nr_steps=100,
        randomise_initial_position=True,
        seed=5,
        nr_cubes=10,
    )
    env = gym.wrappers.ClipAction(env)

    obs, info = env.reset(seed=0)
    done = False
    episode_reward = 0
    t_in_episode = 0

    initial_time = time.time()
    t_max = 10000
    for i in range(t_max):
        if done:
            print(f"Episode finished. Total Reward: {episode_reward:.2f}")
            obs, info = env.reset(seed=0)
            episode_reward = 0
            t_in_episode = 0

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        episode_reward += reward
        t_in_episode += 1

        if i % 30 == 0:
            print(f"Step {t_in_episode} | Action: {action} | Reward: {reward:.2f}")
            print(f"Obs: {obs}")

    elapsed_time = time.time() - initial_time
    print(f"SPS: {t_max / elapsed_time:.2f}")
    env.close()
    print("Environment closed.")
