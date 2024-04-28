#!/bin/sh
env="quartz_physical"
num_landmarks=3
algo="quartz_ppo"
exp="train"
seed=$1

export LD_LIBRARY_PATH=/root/usr_local/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER='PCI_BUS_ID'

echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train/train_quartz.py \
     --use_centralized_V --use_policy_active_masks --num_agents 1 \
     --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
     --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 16 \
     \
     --n_rollout_threads 6 --num_mini_batch 3 --episode_length 600 \
     --num_env_steps 7200000 --ppo_epoch 5 --use_ReLU --gain 0.01 \
     --lr 7e-4 --critic_lr 7e-4 --entropy_coef 0.03 \
     \
     --wandb_name "antonymei" --user_name "antonymei" \
     \
     --reg_degree_types 8 --reg_degree_embedding_dim 64 --gate_is_input_embedding_dim 64 \
     --num_gnn_layers 6 --reg_representation_dim 128 --gate_representation_dim 128 \
     \
     --start_from_internal_prob 0.8 --initial_env_penalty_threshold 150 --initial_env_save_threshold 87 \
     \
     --use_eval --n_eval_rollout_threads 32 --eval_episodes 200 --eval_interval 20 \
     \
     --qasm_file_name "gf2^E5_mult_after_heavy.qasm" --backend_name "IBM_Q65_HUMMINGBIRD" \
     --reversed_qasm_file_name "" \
     --max_obs_length 30000 --eval_max_gate_count 5000 \
     \
     --world_size 4 \
     \
     --search_type "none" --search_interval 100 --search_rounds 20 --round_budget 200 \
     --save_count 10 --random_search_lambda 30 \
     \
     --initial_phase_len 5 --allow_nop_in_initial True --initial_phase_reward 0 \
     \
     --num_sabre_runs 1000 --num_sabre_saves 10 \
     \
     --two_way_mode "none" --two_way_save_interval 100 --two_way_save_count 16 --two_way_clear_on_save True \
     \
     --qasm_save_threshold 1000