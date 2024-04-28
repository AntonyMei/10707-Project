#!/bin/sh
env="quartz_physical"
num_landmarks=3
algo="quartz_ppo"
exp="evaluation"
seed_max=1

export CUDA_DEVICE_ORDER='PCI_BUS_ID'

echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_search_result.py \
     --use_centralized_V --use_policy_active_masks --num_agents 1 \
     --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
     --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 16 \
     \
     --n_rollout_threads 12 --num_mini_batch 1 --episode_length 100 \
     --num_env_steps 500000 --ppo_epoch 10 --use_ReLU --gain 0.01 \
     --lr 7e-4 --critic_lr 7e-4 \
     \
     --wandb_name "antonymei" --user_name "antonymei" \
     \
     --reg_degree_types 8 --reg_degree_embedding_dim 64 --gate_is_input_embedding_dim 64 \
     --num_gnn_layers 6 --reg_representation_dim 128 --gate_representation_dim 128 \
     \
     --start_from_internal_prob 0 --initial_env_penalty_threshold 150 --initial_env_save_threshold 100  \
     \
     --use_eval --n_eval_rollout_threads 32 --eval_episodes 200 \
     \
     --qasm_file_name "gf2^E5_mult_after_heavy.qasm" --backend_name "IBM_Q27_FALCON" \
     --max_obs_length 20000 --eval_max_gate_count 5000 \
     \
     --world_size 4
done