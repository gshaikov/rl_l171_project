#!/bin/bash
strategies=(
    "random"
    "priority_critic"
    "priority_actor"
    "priority_actor_critic"
    "priority_inverse_critic"
    "priority_streaming"
)

for s in "${strategies[@]}"; do
    python -m rl_l171.algos.ddpg \
        --buffer_strategy "$s" \
        --exp_name "ddpg_$s" \
        --exploration_timesteps 25000 \
        --total_timesteps 100000 &
    sleep 1
done
