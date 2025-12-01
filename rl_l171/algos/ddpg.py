# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# necessary when running on headless servers
# to avoid OpenGL error
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import wandb
from gymnasium.wrappers import (
    FlattenObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from rl_l171.algos.buffers import (
    PriorityBuffer,
    PriorityBufferHeap,
    PriorityStreamingBuffer,
    ReplayBuffer,
)
from rl_l171.gym_env import CubesGymEnv

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


VIDEO_ROOT = Path("videos")


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
    wandb_project_name: str = "rl171"
    """the wandb's project name"""
    wandb_entity: str = "leosanitt-university-of-cambridge"
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Cubes-v0"
    """the environment id of the Atari game"""
    total_timesteps: int = 50_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    actor_learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    small_buffer_size: int = int(1e4)
    """the replay memory buffer size for the streaming style algorithm"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 256 * 4
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    max_grad_norm: int = 10

    # exploration
    start_e: float = 2 / 3
    end_e: float = 1 / 30
    # if [0, 1  ] -> interpreted as a fraction
    # if [1, inf] -> interpreted as timesteps
    exploration_timesteps: float = 0.5

    # environment
    max_nr_steps: int = 100
    nr_cubes: int = 5

    """buffer strategy to use"""
    """
    random
    priority_critic
    priority_actor
    priority_actor_critic
    priority_inverse_critic
    priority_streaming
    """
    buffer_strategy: str = "priority_critic"
    pb_beta: float = 0.0
    pb_beta_increment_per_sampling: float = 0.0

    evaluation_frequency: int = 5_000
    """the timesteps between two evaluations"""
    eval_episodes: int = 10
    """the number of episodes to evaluate the agent"""

    capture_video: bool = True


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    env_kwargs: dict | None = None,
    video_trigger: Callable[[int], bool] = lambda i: i == 0,
):
    env_kwargs = dict(
        render_mode=None,
        max_nr_steps=100,
        randomise_initial_position=True,
        seed=seed,
        nr_cubes=5,
    ) | (env_kwargs or {})

    def thunk():
        env = CubesGymEnv(**env_kwargs)
        env.action_space.seed(seed)
        if capture_video:
            env = RecordVideo(
                env,
                VIDEO_ROOT / run_name,
                name_prefix="eval",
                episode_trigger=video_trigger,
            )
        env = RecordEpisodeStatistics(FlattenObservation(env))
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        input_dim = np.prod(env.single_observation_space.shape) + np.prod(
            env.single_action_space.shape
        )
        output_dim = 1

        self.net = nn.Sequential(
            # nn.RMSNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.RMSNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.RMSNorm(256),
            nn.Linear(256, output_dim),
        )
        # self.net.apply(self._sparse_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.net(x)
        return x

    def _sparse_init(self, m):
        if isinstance(m, nn.Linear):
            init.sparse_(m.weight, sparsity=0.9, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()

        input_dim = np.prod(env.single_observation_space.shape)
        output_dim = np.prod(env.single_action_space.shape)

        self.net = nn.Sequential(
            # nn.RMSNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.RMSNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.RMSNorm(256),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

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
        # self.net.apply(self._sparse_init)

    def forward(self, x):
        x = self.net(x)
        return x * self.action_scale + self.action_bias

    def _sparse_init(self, m):
        if isinstance(m, nn.Linear):
            init.sparse_(m.weight, sparsity=0.9, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def linear_schedule(t, t0: int, T: int, x0: float, xT: float):
    if t <= t0:
        return x0
    t -= t0
    T -= t0
    slope = (xT - x0) / T
    return max(slope * t + x0, xT)


@torch.no_grad()
def evaluate(
    make_env: Callable[[], Callable[[], gym.Env]],
    load_model: Callable[[gym.Env, torch.device], nn.Module],
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
):
    envs = gym.vector.SyncVectorEnv([make_env()])
    actor = load_model(envs, device)

    obs, _ = envs.reset()
    episodic_returns = []
    cube_distances = []
    n_cleaned = []
    while len(episodic_returns) < eval_episodes:
        actions = actor(torch.tensor(obs).to(device))
        actions += torch.normal(0, actor.action_scale)
        actions = (
            actions.cpu()
            .numpy()
            .clip(envs.single_action_space.low, envs.single_action_space.high)
        )

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns.append(info["episode"]["r"])
                cube_distances.append(info["cube_distance"])
                n_cleaned.append(info["n_cleaned"])
        obs = next_obs

    envs.close()
    return (
        np.concatenate(episodic_returns),
        np.array(cube_distances),
        np.array(n_cleaned),
    )


def train(wandb_run: "Run"):
    args = wandb_run.config
    run_name = args.run_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                0,
                False,
                run_name,
                env_kwargs={
                    "max_nr_steps": args.max_nr_steps,
                    "nr_cubes": args.nr_cubes,
                },
            )
        ]
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
        list(qf1.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.9),
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.9),
    )

    envs.single_observation_space.dtype = np.float32

    total_reward = 0
    total_square_reward = 0

    match args.buffer_strategy:
        case "random":
            rb = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
            )
        case (
            "priority_critic"
            | "priority_actor"
            | "priority_actor_critic"
            | "priority_inverse_critic"
        ):
            rb = PriorityBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
                beta=args.pb_beta,
                beta_increment_per_sampling=args.pb_beta_increment_per_sampling,
            )
        case "priority_streaming":
            rb = PriorityStreamingBuffer(
                args.small_buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
            )

        case _:
            assert False, "Strategy is invalid"

    wandb_run.define_metric("global_step")
    wandb_run.define_metric("exploration/*", step_metric="global_step")
    wandb_run.define_metric("episode/*", step_metric="global_step")
    wandb_run.define_metric("train/*", step_metric="global_step")

    start_time = time.perf_counter()

    t0 = args.learning_starts
    if 0 <= args.exploration_timesteps <= 1:
        T = t0 + int((args.total_timesteps - t0) * args.exploration_timesteps)
    elif args.exploration_timesteps > 1:
        T = int(args.exploration_timesteps)
    else:
        raise Exception(
            "exploration_timesteps should be in [0, 1] or > 1, "
            f"got {args.exploration_timesteps}"
        )
    epsilon_func = partial(linear_schedule, t0=t0, T=T, x0=args.start_e, xT=args.end_e)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in (pbar := tqdm(range(args.total_timesteps))):
        log = {}
        log.update({"global_step": global_step})

        epsilon = epsilon_func(t=global_step)
        log.update({"exploration/epsilon": epsilon})

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * epsilon)
            actions = (
                actions.cpu()
                .numpy()
                .clip(envs.single_action_space.low, envs.single_action_space.high)
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        rewards *= 10

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    pbar.set_postfix(
                        {
                            "ep_return": info["episode"]["r"],
                            "ep_length": info["episode"]["l"],
                            "ep_cube_dist": info["cube_distance"],
                        }
                    )
                    log.update(
                        {
                            "episode/return": info["episode"]["r"],
                            "episode/length": info["episode"]["l"],
                            "episode/cube_distance": info["cube_distance"],
                            "episode/n_cleaned": info["n_cleaned"],
                        }
                    )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        match args.buffer_strategy:
            case (
                "random"
                | "priority_critic"
                | "priority_actor"
                | "priority_actor_critic"
                | "priority_inverse_critic"
            ):
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
            case "priority_streaming":
                r = abs(rewards.mean().item())
                total_reward += r
                total_square_reward += r**2

                if global_step <= args.learning_starts:
                    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
                else:
                    mean = total_reward / global_step
                    mean_sq = total_square_reward / global_step
                    var = mean_sq - mean**2
                    std = math.sqrt(max(var, 1e-8))
                    z = (r - mean) / std
                    inv_pdf_like = math.exp(0.5 * (z**2))

                    # Map to [0, 1], increasing with rareness
                    p_add = inv_pdf_like / (1.0 + inv_pdf_like)
                    if random.random() < p_add:
                        rb.add(
                            obs, real_next_obs, actions, rewards, terminations, infos
                        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            match args.buffer_strategy:
                case (
                    "random"
                    | "priority_critic"
                    | "priority_inverse_critic"
                    | "priority_actor_critic"
                    | "priority_streaming"
                ):
                    data, btch_ind, weights = rb.sample(
                        args.batch_size, critic_prios=True, actor_prios=False
                    )
                case "priority_actor":
                    data, btch_ind, weights = rb.sample(
                        args.batch_size, critic_prios=False, actor_prios=True
                    )
                case _:
                    assert False, "Strategy is invalid"

            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                target_max = (1 - data.dones.flatten()) * qf1_next_target.view(-1)
                next_q_value = data.rewards.flatten() + args.gamma * target_max

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            td_errors = next_q_value - qf1_a_values
            match args.buffer_strategy:
                case "random" | "priority_actor" | "priority_streaming":
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                case (
                    "priority_critic"
                    | "priority_inverse_critic"
                    | "priority_actor_critic"
                ):
                    weights = torch.as_tensor(
                        weights, device=data.observations.device, dtype=torch.float32
                    ).detach()

                    rb.update_critic_priorities(
                        batch_indices=btch_ind,
                        td_errors=td_errors.detach().cpu().numpy(),  # priority sampling
                        # td_errors=np.array([1] * args.batch_size),  # random sampling
                    )
                    # PER loss: weights multiply the per-sample loss
                    per_sample_loss = F.mse_loss(
                        qf1_a_values, next_q_value, reduction="none"
                    )  # shape [batch]
                    qf1_loss = (weights * per_sample_loss).mean()
                    # qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                case "priority_streaming":
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                case _:
                    assert False, "Strategy is invalid"

            log.update(
                {
                    "train/qf1_values": qf1_a_values.detach().mean().item(),
                    "train/target_max": target_max.detach().mean().item(),
                    "train/reward": data.rewards.flatten().mean().item(),
                    "train/qf1_loss": qf1_loss.detach().item(),
                }
            )

            # optimize the value model
            q_optimizer.zero_grad()
            qf1_loss.backward()

            total_norm_list = []
            for p in qf1.parameters():
                if p.grad is not None:
                    total_norm_list.append(p.grad.detach().norm().item())
            total_norm = torch.tensor(total_norm_list).norm()
            log.update({"train/critic_grad_norm": total_norm})

            clip_grad_norm_(qf1.parameters(), max_norm=args.max_grad_norm)

            q_optimizer.step()

            total_norm_list = []
            for p in qf1.parameters():
                if p.grad is not None:
                    total_norm_list.append(p.data.detach().norm().item())
            total_norm = torch.tensor(total_norm_list).norm()
            log.update({"train/critic_l2_norm": total_norm})

            if (global_step - args.learning_starts) % args.policy_frequency == 0:
                match args.buffer_strategy:
                    case "random" | "priority_critic" | "priority_streaming":
                        pass
                    case (
                        "priority_actor"
                        | "priority_actor_critic"
                        | "priority_inverse_critic"
                    ):
                        data, btch_ind, weights = rb.sample(
                            args.batch_size, critic_prios=False, actor_prios=True
                        )
                        weights = torch.as_tensor(
                            weights,
                            device=data.observations.device,
                            dtype=torch.float32,
                        ).detach()
                    case _:
                        assert False, "Strategy is invalid"

                # optimize the policy model
                actor_td_errors = -qf1(data.observations, actor(data.observations))
                # Ensure weights broadcast correctly
                match args.buffer_strategy:
                    case "priority_critic" | "random":
                        pass
                    case "priority_actor" | "priority_actor_critic":
                        rb.update_actor_priorities(
                            btch_ind, actor_td_errors.detach().cpu().numpy()
                        )
                        weights = weights.view_as(actor_td_errors)
                        # Apply weights (no .detach() unless you explicitly want to break gradients)
                        actor_td_errors = actor_td_errors * weights

                    case "priority_inverse_critic":
                        rb.update_actor_priorities(
                            btch_ind,
                            10
                            - np.minimum(np.abs(td_errors.detach().cpu().numpy()), 10),
                        )
                        weights = weights.view_as(actor_td_errors)
                        # Apply weights (no .detach() unless you explicitly want to break gradients)
                        actor_td_errors = actor_td_errors * weights
                    case "priority_streaming":
                        pass
                    case _:
                        assert False, "Strategy is invalid"

                actor_loss = actor_td_errors.mean()

                log.update({"train/actor_loss": actor_loss.item()})

                actor_optimizer.zero_grad()
                actor_loss.backward()

                total_norm_list = []
                for p in actor.parameters():
                    if p.grad is not None:
                        total_norm_list.append(p.grad.detach().norm().item())
                total_norm = torch.tensor(total_norm_list).norm()
                log.update({"train/actor_grad_norm": total_norm})

                clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)

                actor_optimizer.step()

                total_norm_list = []
                for p in actor.parameters():
                    if p.grad is not None:
                        total_norm_list.append(p.data.detach().norm().item())
                total_norm = torch.tensor(total_norm_list).norm()
                log.update({"train/actor_l2_norm": total_norm})

                # update the target networks
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

        log.update(
            {
                "trains/steps_per_second": int(
                    global_step / (time.perf_counter() - start_time)
                ),
            }
        )

        if (global_step + 1) % args.evaluation_frequency == 0:
            run_name_eval = f"{run_name}-eval-{global_step}"

            episodic_returns, cube_distances, n_cleaned = evaluate(
                make_env=partial(
                    make_env,
                    env_id=args.env_id,
                    seed=args.seed + global_step,
                    idx=0,
                    capture_video=args.capture_video and (global_step + 1) == args.total_timesteps,
                    run_name=run_name_eval,
                    env_kwargs={
                        "render_mode": "rgb_array" if args.capture_video else None,
                        "max_nr_steps": 500,
                        "nr_cubes": args.nr_cubes,
                    },
                ),
                load_model=lambda _, device: actor.to(device),
                eval_episodes=args.eval_episodes,
                device=device,
            )

            log.update(
                {
                    "eval/return_mean": episodic_returns.mean().item(),
                    "eval/return_std": episodic_returns.std().item(),
                    "eval/cube_distance_mean": cube_distances.mean().item(),
                    "eval/cube_distance_std": cube_distances.std().item(),
                    "eval/n_cleaned_mean": n_cleaned.mean().item(),
                    "eval/n_cleaned_std": n_cleaned.std().item(),
                }
            )

            video_dir = VIDEO_ROOT / run_name_eval
            for vid_file in video_dir.glob("*.mp4"):
                log.update(
                    {
                        f"eval/video_{vid_file.stem}": wandb.Video(
                            str(vid_file), caption=vid_file.stem, fps=30, format="mp4"
                        )
                    }
                )

        wandb_run.log(log)

    if args.save_model:
        model_dir = Path("runs") / run_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    import tyro

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.monotonic()}"

    assert args.total_timesteps % args.evaluation_frequency == 0

    with wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=dict(**asdict(args), run_name=run_name),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    ) as wandb_run:
        train(wandb_run)
