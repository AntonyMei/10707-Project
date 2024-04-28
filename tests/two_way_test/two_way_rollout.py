#!/usr/bin/env python
from quartz import PySimpleHybridEnv
import multiprocessing as mp
import os
import shutil
import tqdm
import socket
import sys
import time
from pathlib import Path

import SMOS
import numpy as np
import torch
import wandb
import argparse
import gym

from gym import spaces
from onpolicy.algorithms.quartz_ppo_dual.quartz_ppo_model import QuartzPPOModel as Policy
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.scripts.train.train_quartz import parse_args
from onpolicy.utils.quartz_utils import restore_observation, flatten_action, flatten_observation, restore_action


class RolloutEnv(gym.Env):
    def __init__(self, qasm_file_name, backend_name, all_args, env_seed, initial_mapping_file_path,
                 max_obs_length=2000, is_eval=False):
        # prepare underlying env
        start_from_internal_prob = 0 if is_eval else all_args.start_from_internal_prob
        self.env = PySimpleHybridEnv(
            # basic parameters
            qasm_file_path=qasm_file_name,
            backend_type_str=backend_name,
            initial_mapping_file_path=initial_mapping_file_path,
            # randomness and buffer
            seed=env_seed,
            start_from_internal_prob=start_from_internal_prob,
            # GameHybrid settings
            initial_phase_len=all_args.initial_phase_len,
            allow_nop_in_initial=all_args.allow_nop_in_initial,
            initial_phase_reward=all_args.initial_phase_reward,
        )

        # infer environment statistics (space size)
        state = self.env.get_state()
        num_registers = len(state.physical2logical_mapping)
        self._action_space_size = num_registers
        self._max_observation_length = max_obs_length

        # prepare spaces
        # Note: 1. observation space is flattened by flatten_observation
        #       2. shared_observation_space is redundant
        self.action_space = [spaces.Discrete(num_registers * num_registers)]
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.Inf, dtype=int,
                                             shape=(self._max_observation_length,))]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.Inf, dtype=int,
                                                   shape=(self._max_observation_length,))]

        # data collecting statistics
        self.best_total_cost = all_args.eval_max_gate_count

        # record current mapping
        self.current_mapping = None

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action):
        # action here is a number that needs decoding
        # apply action and get reward & check finished
        qubit_idx_0, qubit_idx_1 = restore_action(action)
        reward = [[self.env.step_with_id(qubit_idx_0, qubit_idx_1)]]
        done = [self.env.is_finished()]

        # get next obs
        if done[0]:
            next_obs = None
        else:
            next_state = self.env.get_state()
            action_space = self.env.get_action_space()
            flattened_obs = flatten_observation(graph_state=next_state.circuit,
                                                device_edge_list=next_state.device_edges_list,
                                                logical2physical=next_state.logical2physical_mapping,
                                                physical2logical=next_state.physical2logical_mapping,
                                                action_space=action_space,
                                                is_initial_phase=next_state.is_initial_phase,
                                                target_length=self._max_observation_length, )
            next_obs = [flattened_obs]

            # for record
            self.current_mapping = next_state.logical2physical_mapping

        # put logging items in info
        cur_total_cost = -1
        final_mapping = None
        if done[0]:
            cur_total_cost = self.env.total_cost()
            self.best_total_cost = min(self.best_total_cost, cur_total_cost)
            final_mapping = self.current_mapping

            # save the result
            time_stamp = int(time.time())
            self.env.save_context(
                execution_history_file_path=f"./results/{cur_total_cost}_{time_stamp}_execution_history.txt",
                single_qubit_gate_execution_plan_file_path=f"./results/{cur_total_cost}_{time_stamp}_single.txt"
            )
            self.env.generate_mapped_qasm(
                mapped_qasm_file_path=f"./results/{cur_total_cost}_{time_stamp}_mapped.qasm",
                debug_mode=True
            )

        info = [{"individual_reward": reward[0][0], "best_total_cost": self.best_total_cost,
                 "cur_total_cost": cur_total_cost, "final_mapping": final_mapping}]

        return next_obs, reward, done, info

    def reset(self):
        # reset environment
        self.env.reset()

        # return new observation
        next_state = self.env.get_state()
        action_space = self.env.get_action_space()
        flattened_obs = flatten_observation(graph_state=next_state.circuit,
                                            device_edge_list=next_state.device_edges_list,
                                            logical2physical=next_state.logical2physical_mapping,
                                            physical2logical=next_state.physical2logical_mapping,
                                            action_space=action_space,
                                            is_initial_phase=next_state.is_initial_phase,
                                            target_length=self._max_observation_length, )
        next_obs = [flattened_obs]

        return next_obs


def make_rollout_env(all_args, qasm_file_name, backend_name, initial_mapping_file_path,
                     seed, parallel_rounds):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "quartz_physical":
                env = RolloutEnv(qasm_file_name="../../onpolicy/scripts/qasm_files/" + qasm_file_name,
                                 backend_name=backend_name,
                                 all_args=all_args,
                                 env_seed=seed + rank * 10 + 100000,
                                 initial_mapping_file_path=initial_mapping_file_path,
                                 max_obs_length=all_args.max_obs_length,
                                 is_eval=True)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(seed + rank * 10 + 100000)
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(parallel_rounds)])


class QuartzEvaluator:
    def __init__(self, rank, all_args, round_number):
        # store input parameters
        self.rank = rank
        self.all_args = all_args
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.eval_episode_length = self.all_args.eval_episodes
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = round_number
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # initialize environments and models
        self.policy = None
        self.eval_envs = None
        self.current_model_step = 0

    def evaluate(self, deterministic):
        # initialize
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        self.policy.actor_critic.eval()

        # prepare an array for results
        eval_total_cost = np.ones(self.n_eval_rollout_threads, dtype=int) * self.all_args.eval_max_gate_count
        eval_terminate_step = np.ones(self.n_eval_rollout_threads, dtype=int) * self.eval_episode_length
        finished_count = 0

        # start evaluation
        final_logical2physical_mapping_list = []
        for _cur_step in tqdm.tqdm(range(self.eval_episode_length)):
            # quartz: reconstruct obs
            circuit_batch, device_edge_list_batch, logical2physical_mapping_batch, \
                physical2logical_mapping_batch, action_space_batch = [], [], [], [], []
            for obs in np.array(list(eval_obs[:, 0])):
                circuit, device_edge_list, logical2physical, physical2logical_mapping, action_space, is_initial_phase \
                    = restore_observation(obs)
                circuit_batch.append(circuit)
                device_edge_list_batch.append(device_edge_list)
                logical2physical_mapping_batch.append(logical2physical)
                physical2logical_mapping_batch.append(physical2logical_mapping)
                action_space_batch.append(action_space)

            eval_action = self.policy.act(circuit_batch=circuit_batch,
                                          device_edge_list_batch=device_edge_list_batch,
                                          physical2logical_mapping_batch=physical2logical_mapping_batch,
                                          logical2physical_mapping_batch=logical2physical_mapping_batch,
                                          action_space_batch=action_space_batch,
                                          deterministic=deterministic)
            eval_action = [flatten_action(_action[0], _action[1]) for _action in eval_action]
            eval_actions_env = eval_action

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # store evaluation final result if finished
            for idx, zip_pack in enumerate(zip(eval_dones, eval_infos)):
                done, info = zip_pack[0][0], zip_pack[1]
                assert type(done) == np.bool_ or type(done) == bool
                assert "cur_total_cost" in info[0].keys()
                # check that this is the first time the env is finished in this eval round
                if done and eval_total_cost[idx] == self.all_args.eval_max_gate_count:
                    eval_total_cost[idx] = info[0]['cur_total_cost']
                    final_logical2physical_mapping_list.append((info[0]['cur_total_cost'], info[0]['final_mapping']))
                    eval_terminate_step[idx] = _cur_step
                    finished_count += 1
            if finished_count == self.n_eval_rollout_threads:
                break
        # check that we have received all mappings
        # assert len(final_logical2physical_mapping_list) == self.n_eval_rollout_threads, \
        #     f"{len(final_logical2physical_mapping_list)} != {self.n_eval_rollout_threads}!"

        # get summarized result
        min_total_cost = min(eval_total_cost)
        max_total_cost = max(eval_total_cost)
        avg_total_cost = sum(eval_total_cost) / len(eval_total_cost)

        # collect final results
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_total_reward = 0
        for idx, valid_step_count in enumerate(eval_terminate_step):
            eval_total_reward += np.sum(eval_episode_rewards[0: valid_step_count + 1, idx, 0])
        eval_average_episode_rewards = eval_total_reward / self.n_eval_rollout_threads
        eval_train_infos = {'eval_average_episode_rewards': eval_average_episode_rewards,
                            'eval_min_total_cost': min_total_cost,
                            'eval_max_total_cost': max_total_cost,
                            'eval_avg_total_cost': avg_total_cost,
                            'final_mappings': final_logical2physical_mapping_list}

        # return final result
        return eval_train_infos


def rollout_cost_metric(all_args, qasm_file_name, device_name, initial_mapping_file_path,
                        model_path, round_number, seed, deterministic=True, cuda_idx=0):
    # initialize evaluator
    evaluator = QuartzEvaluator(rank=cuda_idx, all_args=all_args, round_number=round_number)
    evaluator.eval_envs = make_rollout_env(all_args=all_args,
                                           qasm_file_name=qasm_file_name,
                                           backend_name=device_name,
                                           initial_mapping_file_path=initial_mapping_file_path,
                                           parallel_rounds=round_number,
                                           seed=seed)
    evaluator.policy = Policy(reg_degree_types=all_args.reg_degree_types,
                              reg_degree_embedding_dim=all_args.reg_degree_embedding_dim,
                              gate_is_input_embedding_dim=all_args.gate_is_input_embedding_dim,
                              num_gnn_layers=all_args.num_gnn_layers,
                              reg_representation_dim=all_args.reg_representation_dim,
                              gate_representation_dim=all_args.gate_representation_dim,
                              # optimization process
                              device=torch.device(f"cuda:{cuda_idx}"),
                              rank=None,  # set to None so that DDP is not enabled for evaluation
                              lr=all_args.lr,
                              allow_nop=all_args.allow_nop_in_initial,
                              opti_eps=all_args.opti_eps,
                              weight_decay=all_args.weight_decay)

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load model
    raw_state_dict = torch.load(model_path)
    state_dict = {'.'.join([k.split('.')[0]] + k.split('.')[2:]): v.cpu() for k, v in raw_state_dict.items()}
    evaluator.policy.actor_critic.load_state_dict(state_dict=state_dict)

    # evaluate and return result
    result = evaluator.evaluate(deterministic=deterministic)
    eval_average_episode_rewards = result['eval_average_episode_rewards']
    eval_min_total_cost = result['eval_min_total_cost']
    eval_avg_total_cost = result['eval_avg_total_cost']
    eval_max_total_cost = result['eval_max_total_cost']
    final_mappings = result['final_mappings']

    # clean up and return
    evaluator.eval_envs.close()
    return eval_average_episode_rewards, eval_min_total_cost, eval_avg_total_cost, eval_max_total_cost, final_mappings


def test_two_way_rollout(args):
    # parse arguments
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.algorithm_name == "quartz_ppo"

    # ------------------ some more parameters ------------------ #
    forward_qasm_file_name = "gf2^E5_mult_after_heavy.qasm"
    backward_qasm_file_name = "gf2^E5_mult_after_heavy_reversed.qasm"
    device_name = "IBM_Q65_HUMMINGBIRD"
    forward_model_id = "1"
    backward_model_id = "1"
    parallel_env_number = 128
    save_count = 128
    round_length = 10000
    seed = all_args.seed
    deterministic = False
    start_from_forward = False
    cuda_idx = 3
    # ---------------------------------------------------------- #
    # print parameters
    print("forward_qasm_file_name:", forward_qasm_file_name)
    print("backward_qasm_file_name:", backward_qasm_file_name)
    print("device_name:", device_name)
    print("forward_model_id:", forward_model_id)
    print("backward_model_id:", backward_model_id)
    print("parallel_env_number:", parallel_env_number)
    print("save_count:", save_count)
    print("round_length:", round_length)
    print("seed:", seed)
    print("deterministic:", deterministic)
    print("start_from_forward:", start_from_forward)
    print("cuda_idx:", cuda_idx)

    # prepare initial working environment
    working_dir = f"./experiment/{int(time.time())}/"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(f"./results", exist_ok=True)
    if start_from_forward:
        shutil.copyfile(src=f"./model/forward_{forward_model_id}/mapping.txt",
                        dst=working_dir + "forward_mapping_0.txt")
    else:
        shutil.copyfile(src=f"./model/backward_{backward_model_id}/mapping.txt",
                        dst=working_dir + "backward_mapping_0.txt")

    # some logging items
    global_best_total_cost = 100000
    global_best_step_idx = None

    # rollout
    is_forward = start_from_forward
    for step_idx in range(round_length):
        print(f"Pass {step_idx}: is_forward={is_forward}")

        # prepare this step's env parameters
        if is_forward:
            initial_mapping_file_path = working_dir + f"forward_mapping_{step_idx}.txt"
            model_path = f"./model/forward_{forward_model_id}/model_forward_{forward_model_id}.pt"
            qasm_file_name = forward_qasm_file_name
        else:
            initial_mapping_file_path = working_dir + f"backward_mapping_{step_idx}.txt"
            model_path = f"./model/backward_{backward_model_id}/model_backward_{backward_model_id}.pt"
            qasm_file_name = backward_qasm_file_name

        # run evaluation
        # final_mappings: a list of (cost, mapping)
        eval_average_episode_rewards, eval_min_total_cost, eval_avg_total_cost, eval_max_total_cost, final_mappings = \
            rollout_cost_metric(all_args=all_args, qasm_file_name=qasm_file_name, device_name=device_name,
                                initial_mapping_file_path=initial_mapping_file_path, model_path=model_path,
                                round_number=parallel_env_number, seed=seed, deterministic=deterministic,
                                cuda_idx=cuda_idx)

        # save mappings for next round (only save best k mappings)
        is_forward = not is_forward
        # file name
        if is_forward:
            new_mapping_file_name = working_dir + f"forward_mapping_{step_idx + 1}.txt"
        else:
            new_mapping_file_name = working_dir + f"backward_mapping_{step_idx + 1}.txt"
        # pick the mappings to save
        mappings_to_save = sorted(final_mappings, key=lambda tup: tup[0])[0: save_count]
        worst_saved_total_cost = mappings_to_save[-1][0]
        # save
        with open(new_mapping_file_name, "w") as file:
            for cost_mapping_tuple in mappings_to_save:
                mapping = cost_mapping_tuple[1]
                for idx in range(len(mapping)):
                    file.write(f"{mapping[idx]} ")
                file.write("\n")

        # log some statistics
        if eval_min_total_cost <= global_best_total_cost:
            global_best_total_cost = eval_min_total_cost
            global_best_step_idx = step_idx
        print(f"Global best total cost: {global_best_total_cost} "
              f"(round_id = {global_best_step_idx},"
              f" is_forward={global_best_step_idx % 2 == 0 ^ (not start_from_forward)}).")
        print(f"eval_average_episode_rewards: {eval_average_episode_rewards}.")
        print(f"eval_min_total_cost: {eval_min_total_cost}.")
        print(f"eval_avg_total_cost: {eval_avg_total_cost}.")
        print(f"eval_max_total_cost: {eval_max_total_cost}.")
        print(f"Saved {save_count} mappings, with worst cost {worst_saved_total_cost}.\n")


if __name__ == '__main__':
    test_two_way_rollout(sys.argv[1:])
