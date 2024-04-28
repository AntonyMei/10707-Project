#!/bin/sh
env="quartz_physical"
num_landmarks=3
algo="quartz_ppo"
exp="Scalability"
seed=$1
device_name=$2
qasm_file_name=$3
reversed_qasm_file_name=$4

export LD_LIBRARY_PATH=~/usr_local/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER='PCI_BUS_ID'

echo "Info: env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo "Info: this is backward agent, experiment circuit is ${reversed_qasm_file_name} (\"reversed\" should not appear in the name)!"
echo "Info: device is ${device_name}, qasm_file_name is ${qasm_file_name}, reversed_qasm_file_name is ${reversed_qasm_file_name}"

    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train/train_quartz.py \
     --use_centralized_V --use_policy_active_masks --num_agents 1 \
     --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
     --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 16 \
     \
     --n_rollout_threads 3 --num_mini_batch 1 --episode_length 200 \
     --num_env_steps 60000000 --ppo_epoch 5 --use_ReLU --gain 0.01 \
     --lr 7e-4 --critic_lr 7e-4 --entropy_coef 0.03 \
     \
     --wandb_name "antonymei" --user_name "antonymei" \
     \
     --reg_degree_types 8 --reg_degree_embedding_dim 64 --gate_is_input_embedding_dim 64 \
     --num_gnn_layers 6 --reg_representation_dim 128 --gate_representation_dim 128 \
     \
     --start_from_internal_prob 0.2 --game_buffer_size 200 --game_buffer_save_interval 5 \
     \
     --initial_env_penalty_threshold 150 --initial_env_save_threshold 87 \
     \
     --n_eval_rollout_threads 32 --eval_episodes 200 --eval_interval 20 \
     \
     --qasm_file_name "${qasm_file_name}" --backend_name "${device_name}" \
     --reversed_qasm_file_name "${reversed_qasm_file_name}" \
     --max_obs_length 30000 --eval_max_gate_count 30000 \
     \
     --world_size 4 \
     \
     --search_type "none" --search_interval 100 --search_rounds 20 --round_budget 200 \
     --save_count 10 --random_search_lambda 30 \
     \
     --initial_phase_len 5 --allow_nop_in_initial --initial_phase_reward 0 \
     \
     --num_sabre_runs 100 --num_sabre_saves 10 \
     \
     --two_way_mode "backward" --two_way_save_interval 100 --two_way_save_count 8 --two_way_clear_on_save \
     --two_way_start_epoch 3600 \
     \
     --qasm_save_threshold 40000 \
     \
     --pretrain_mode "full"