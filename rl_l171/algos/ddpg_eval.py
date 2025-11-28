"""
python3 -m rl_l171.algos.ddpg_eval
"""

from functools import partial
from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn

from rl_l171.algos.ddpg import Actor, make_env, Args, VIDEO_ROOT, evaluate


def load_model(
    model_path: str,
    Model: type[nn.Module],
    envs: gym.Env,
    device: torch.device,
):
    actor = Model(envs).to(device)
    actor_params, _ = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    return actor


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

    episodic_returns, cube_distances, n_cleaned = evaluate(
        make_env=partial(
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
        load_model=partial(load_model, model_path=model_path, Model=Actor),
        eval_episodes=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    log = {}
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
