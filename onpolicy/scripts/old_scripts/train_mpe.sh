#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=3
num_agents=3
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --share_policy \
     --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}\
     --scenario_name ${scenario} --num_agents ${num_agents} \
     --num_landmarks ${num_landmarks} --seed ${seed} \
     --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 \
     --episode_length 25 --num_env_steps 20000000 --ppo_epoch 10 \
     --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
     --use_recurrent_policy --wandb_name "antonymei" --user_name "antonymei" \
     --device_out_dimension 64 --circuit_num_layers 6 --gate_type_embedding_dim 64 \
     --circuit_conv_internal_dim 128 --circuit_out_dimension 128 \
     --final_mlp_hidden_dimension_ratio 4 --num_attention_heads 8 \
     --attention_qk_dimension 64 --attention_v_dimension 64 \
     --num_registers 20
done