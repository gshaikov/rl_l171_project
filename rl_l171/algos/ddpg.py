# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import (
    FlattenObservation,
    RecordEpisodeStatistics,
)  # or from gymnasium.wrappers import FlattenObservation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from rl_l171.algos.buffers import ReplayBuffer, PriorityBufferHeap, PriorityBuffer
from rl_l171.algos.dqn import linear_schedule
from rl_l171.gym_env import CubesGymEnv

n_cubes = 5
cube_env = CubesGymEnv(
    render_mode="None",
    max_nr_steps=100,
    randomise_initial_position=True,
    seed=5,
    nr_cubes=n_cubes,
)

cube_env_render = CubesGymEnv(
    render_mode="human",
    max_nr_steps=100,
    randomise_initial_position=True,
    seed=5,
    nr_cubes=n_cubes,
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
    wandb_entity: str = "leosanitt-university-of-cambridge"
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
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    max_grad_norm: int = 10


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
        env = RecordEpisodeStatistics(FlattenObservation(env))

        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video and idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env.seed(seed)
        return env

    return thunk


def make_env_render(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        #     env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # env.action_space.seed(seed)
        #
        env = cube_env_render
        # 👇 flatten Dict -> Box (and Box stays Box)
        env = RecordEpisodeStatistics(FlattenObservation(env))

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
        # self.norm0 = nn.LayerNorm(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        # self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        # self.norm2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # self.norm0 = nn.LayerNorm(np.array(env.single_observation_space.shape).prod())
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        # self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        # self.norm2 = nn.LayerNorm(256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
    import wandb
    from tqdm import tqdm

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if not args.track:
        raise AttributeError("must use --track.")
    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()), lr=args.learning_rate, weight_decay=1e-4
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()), lr=args.learning_rate, weight_decay=1e-4
    )

    envs.single_observation_space.dtype = np.float32
    rb = PriorityBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    wandb_run.define_metric("global_step")
    wandb_run.define_metric("exploration/*", step_metric="global_step")
    wandb_run.define_metric("episode/*", step_metric="global_step")
    wandb_run.define_metric("train/*", step_metric="global_step")

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in (pbar := tqdm(range(args.total_timesteps))):
        log = {}
        log.update({"global_step": global_step})

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(envs.single_action_space.low, envs.single_action_space.high)
                )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        state_value = envs.get_attr("state_value")[0]
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            log.update({"episode/return": rewards.mean()})
            log.update({"episode/cube_distance": -state_value})
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
            data, btch_ind = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                target_max = (1 - data.dones.flatten()) * qf1_next_target.view(-1)
                next_q_value = data.rewards.flatten() + args.gamma * target_max

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            td_errors = next_q_value - qf1_a_values
            rb.update_priorities(
                batch_indices=btch_ind,
                td_errors=td_errors.detach().cpu().numpy(),  # or td_errors.detach()
            )

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()

            total_norm = 0.0
            for p in qf1.parameters():
                total_norm += p.grad.norm(2).detach().pow(2).item()
            total_norm = torch.tensor(total_norm).sqrt()

            log.update({"train/critic_grad_norm": total_norm})

            critic_grad_norm = clip_grad_norm_(
                qf1.parameters(), max_norm=args.max_grad_norm
            )

            q_optimizer.step()

            total_norm = torch.sqrt(
                sum(p.data.pow(2).sum() for p in qf1.parameters() if p.grad is not None)
            ).item()
            log.update({"train/critic_l2_norm": total_norm})
            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()

                total_norm = 0.0
                for p in actor.parameters():
                    total_norm += p.grad.norm(2).detach().pow(2).item()
                total_norm = torch.tensor(total_norm).sqrt()

                log.update({"train/actor_grad_norm": total_norm})

                actor_grad_norm = clip_grad_norm_(
                    actor.parameters(), max_norm=args.max_grad_norm
                )

                actor_optimizer.step()
                total_norm = torch.sqrt(
                    sum(
                        p.data.pow(2).sum()
                        for p in actor.parameters()
                        if p.grad is not None
                    )
                ).item()
                log.update({"train/actor_l2_norm": total_norm})

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                # if global_step % 100 == 0:
                log.update(
                    {
                        "train/qf1_values": qf1_a_values.mean().item(),
                        "train/qf1_loss": qf1_loss.item(),
                        "train/actor_loss": actor_loss.item(),
                        "train/target_max": target_max.mean().item(),
                        "train/reward": data.rewards.flatten().mean().item(),
                        "trains/steps_per_second": int(
                            global_step / (time.time() - start_time)
                        ),
                    }
                )
        wandb_run.log(log)

    if args.save_model:
        model_dir = Path("runs") / run_name
        model_dir.mkdir(parents=True, exist_ok=True)  # <-- ensure directory exists

        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        from rl_l171.algos.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env_render,
            args.env_id,
            eval_episodes=1,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=0,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            log.update({"episode/return": episodic_return}, step=idx)
            wandb_run.log(log)
    envs.close()
    wandb.finish()
