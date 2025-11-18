import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_l171.gym_env import CubesGymEnv

env = CubesGymEnv(render_mode="None", max_nr_steps=100, randomise_initial_position=True,
                  seed=5, nr_cubes=10)

env = gym.wrappers.ClipAction(env)
class RandomAgent:
    """
    An agent that selects actions uniformly at random from the environment's action space.
    """
    def __init__(self, action_space: spaces.Box):
        """
        Initialize the RandomAgent with the environment's action space.

        Args:
            action_space: The gymnasium.spaces.Box object defining the action limits.
        """
        self.action_space = action_space

    def select_action(self, observation: dict) -> np.ndarray:
        """
        Selects a random action.

        Args:
            observation: The observation dict received from the environment (not used here).

        Returns:
            A numpy array representing the random action.
        """
        # The action space is a Box, which has a convenient sample() method
        return self.action_space.sample()

    def train(self) -> None:
        pass

if __name__ == '__main__':
    obs, info = env.reset(seed=0)
    done = False
    episode_reward = 0
    t_in_episode = 0
    agent = RandomAgent(env.action_space)

    initial_time = time.time()
    t_max = 1000
    for i in range(t_max):
        if done:
            print(f"Episode finished. Total Reward: {episode_reward:.2f}")
            obs, info = env.reset(seed=0)
            episode_reward = 0
            t_in_episode = 0

        action = agent.select_action(obs)

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