# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers import (
    FlattenObservation,
)  # or from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from rl_l171.algos.buffers import ReplayBuffer
from rl_l171.gym_env import CubesGymEnv

cube_env = CubesGymEnv(
    render_mode="None",
    max_nr_steps=100,
    randomise_initial_position=True,
    seed=5,
    nr_cubes=10,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl171"
    """the wandb's project name"""
    wandb_entity: str | None = "leosanitt-university-of-cambridge"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Cubes-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.action_space.seed(seed)
        #
        env = cube_env
        # 👇 flatten Dict -> Box (and Box stays Box)
        env = FlattenObservation(env)

        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(np.array(env.single_observation_space.shape).prod()),
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.LayerNorm(120),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.LayerNorm(84),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import wandb
    from tqdm import tqdm

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if not args.track:
        raise AttributeError("must use --track.")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    wandb_run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    wandb_run.define_metric("global_step")
    wandb_run.define_metric("exploration/*", step_metric="global_step")
    wandb_run.define_metric("episode/*", step_metric="global_step")
    wandb_run.define_metric("train/*", step_metric="global_step")

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.perf_counter()

    q_network.eval()
    target_network.eval()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in (pbar := tqdm(range(args.total_timesteps))):
        log = {}
        log.update({"global_step": global_step})

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        log.update({"exploration/epsilon": epsilon})

        with torch.no_grad():
            q_values = q_network(
                torch.tensor(obs).to(device=device, dtype=q_network.dtype)
            ).cpu()

        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions = torch.argmax(q_values, dim=-1).numpy()

        log.update({"exploration/q_values": q_values[0][actions[0]].item()})

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    pbar.set_postfix(
                        {
                            "ep_return": info["episode"]["r"],
                            "ep_length": info["episode"]["l"],
                        }
                    )
                    log.update(
                        {
                            "episode/return": info["episode"]["r"],
                            "episode/length": info["episode"]["l"],
                        }
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if (global_step - args.learning_starts) % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(
                        data.next_observations.to(
                            device=device, dtype=target_network.dtype
                        )
                    ).max(dim=-1)
                    target_max = target_max * (1 - data.dones.flatten().int())
                    replay_reward = data.rewards.flatten()
                    td_target = replay_reward + args.gamma * target_max

                q_network.train()
                old_val = (
                    q_network(
                        data.observations.to(device=device, dtype=q_network.dtype)
                    )
                    .gather(1, data.actions)
                    .squeeze()
                )

                loss = F.mse_loss(old_val, td_target)

                log.update(
                    {
                        "train/loss": loss.detach().cpu(),
                        "train/q_values": old_val.detach().cpu().mean().item(),
                        "train/target_max": target_max.cpu().mean().item(),
                        "train/reward": replay_reward.cpu().mean().item(),
                        "train/td_target": td_target.cpu().mean().item(),
                        # "train/lr": lr_scheduler.get_last_lr()[0],
                    }
                )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()

                total_norm = 0.0
                for p in q_network.parameters():
                    total_norm += p.grad.norm(2).detach().pow(2).item()
                total_norm = torch.tensor(total_norm).sqrt()

                log.update({"train/grad_norm": total_norm})

                optimizer.step()
                q_network.eval()

                dt = global_step / (time.perf_counter() - start_time)
                log.update({"train/steps_per_second": dt})

            # update target network
            if (
                global_step - args.learning_starts
            ) % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )

        wandb_run.log(log)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from rl_l171.algos.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            print(f"eval_episode={idx}, episodic_return={episodic_return}")
            # wandb_run.log(
            #     {
            #         f"eval/episode_return_{idx}": episodic_return,
            #         # "eval/mean_returns": np.mean(episodic_return).item(),
            #         # "eval/std_returns": np.std(episodic_return).item(),
            #     }
            # )

    envs.close()
    wandb_run.finish()

"""
python3 -m rl_l171.algos.dqn --track --total_timesteps 10000
"""
