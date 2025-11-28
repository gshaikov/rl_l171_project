"""
python3 -m rl_l171.algos.ddpg_eval
"""

from functools import partial
from typing import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl_l171.algos.ddpg import Actor, QNetwork, make_env, Args, VIDEO_ROOT


def evaluate(
    model_path: str,
    make_env: Callable,
    eval_episodes: int,
    Model: tuple[type[nn.Module], type[nn.Module]],
    device: torch.device = torch.device("cpu"),
    exploration_noise: float = 0,
):
    envs = gym.vector.SyncVectorEnv([make_env()])
    actor = Model[0](envs).to(device)
    qf = Model[1](envs).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()
    # note: qf is not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
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
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns.append(info["episode"]["r"])
        obs = next_obs

    envs.close()
    return np.concatenate(episodic_returns)


@dataclass
class EvalArgs(Args):
    seed: int = 42
    wandb_project_name: str = "rl171_eval"
    run_name: str = "Cubes-v0__ddpg__1__1764252577"


if __name__ == "__main__":
    import wandb
    import tyro

    args = tyro.cli(EvalArgs)

    model_path = f"runs/{args.run_name}/{args.exp_name}.cleanrl_model"

    wandb_run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.run_name,
        monitor_gym=True,
        save_code=True,
    )

    run_name_eval = f"{args.run_name}-eval"
    video_dir = VIDEO_ROOT / run_name_eval

    episodic_returns = evaluate(
        model_path,
        partial(
            make_env,
            env_id=args.env_id,
            seed=args.seed,
            idx=0,
            capture_video=True,
            run_name=run_name_eval,
            env_kwargs={
                "render_mode": "rgb_array",
                "max_nr_steps": 500,
                "nr_cubes": args.nr_cubes,
            },
            video_trigger=lambda _: True,
        ),
        eval_episodes=4,
        Model=(Actor, QNetwork),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        exploration_noise=0,
    )

    log = {}
    log.update(
        {
            "eval/return_mean": episodic_returns.mean().item(),
            "eval/return_std": episodic_returns.std().item(),
        }
    )
    wandb_run.log(log)

    for vid_file in video_dir.glob("*.mp4"):
        wandb_run.log(
            {
                f"eval/video_{vid_file.stem}": wandb.Video(
                    str(vid_file), caption=vid_file.stem, fps=30, format="mp4"
                )
            }
        )

    wandb_run.finish()
