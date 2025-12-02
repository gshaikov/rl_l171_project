import time
from dataclasses import asdict, dataclass

import numpy as np

from rl_l171.algos.ddpg import Args


@dataclass
class SweepArgs(Args):
    wandb_project_name: str = "rl171_sweep"
    wandb_sweep_id: str | None = None
    seed: int = 0
    method: str = "random"
    metric: str = "eval/cube_distance_mean"
    goal: str = "minimize"

    # latest sweep
    total_timesteps: int = 200_000
    eval_episodes: int = 10
    evaluation_frequency: int = 20_000


def to_sweep(args: SweepArgs) -> dict:
    cfg_params = {}
    for k, v in asdict(args).items():
        if k in ["wandb_project_name", "wandb_entity", "wandb_sweep_id"]:
            continue
        cfg_params[k] = {"value": v}

    cfg_params.update(
        {
            "seed": {"values": [0, 1, 2, 3, 4]},
            "learning_rate": {"values": np.linspace(1e-4, 1e-3, num=4).round(4).tolist()},
            "buffer_size": {"values": [10_000, 1_000_000]},
            "tau": {"values": np.linspace(1e-3, 1e-2, num=4).round(4).tolist()},
            "batch_size": {"values": [256, 512, 1024]},
            "learning_starts": {"values": [2048, 4096]},
            "policy_frequency": {"values": [2, 4]},
            "max_grad_norm": {"values": [1.0, 10.0]},
            "exploration_timesteps": {"values": [0.25, 0.5, 0.75]},
            "max_nr_steps": {"values": [100, 200]},
            "buffer_strategy": {
                "values": [
                    "random",
                    "priority_critic",
                    "priority_actor",
                    "priority_actor_critic",
                    "priority_inverse_critic",
                    # "priority_streaming",  # does not work yet
                ]
            },
        }
    )

    cfg_sweep = {
        "name": f"ddpg_sweep_{int(time.monotonic())}",
        "method": args.method,
        "metric": {"goal": args.goal, "name": args.metric},
        "parameters": cfg_params,
    }
    return cfg_sweep


if __name__ == "__main__":
    import multiprocessing

    import tyro
    import wandb

    from rl_l171.algos.ddpg import train

    args = tyro.cli(SweepArgs)

    run_name = f"{args.env_id}__{args.exp_name}__{time.monotonic()}"

    def main():
        with wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=dict(**asdict(args), run_name=run_name),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        ) as wandb_run:
            train(wandb_run)

    if args.wandb_sweep_id is None:
        print("Starting a new sweep...")
        sweep_id = wandb.sweep(
            sweep=to_sweep(args),
            entity=args.wandb_entity,
            project=args.wandb_project_name,
        )
    else:
        sweep_id = args.wandb_sweep_id
        print(f"Using existing sweep: {sweep_id}")

    num_workers = 8
    runs_per_worker = 24

    def run_agent():
        wandb.agent(
            sweep_id,
            function=main,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            count=runs_per_worker,
        )

    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=run_agent)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("All processes finished.")
