# this must be the first import
from quartz import PySimpleHybridEnv

import time
import gym
import numpy as np

from gym import spaces
from onpolicy.utils.quartz_utils import flatten_observation, restore_action


class SimpleHybridEnv(gym.Env):
    def __init__(self, qasm_file_name, backend_name, all_args, env_seed, ddp_rank, save_dir, save_threshold,
                 max_obs_length=2000, is_eval=False):
        # prepare underlying env
        start_from_internal_prob = 0 if is_eval else all_args.start_from_internal_prob
        self.env = PySimpleHybridEnv(
            # basic parameters
            qasm_file_path=qasm_file_name,
            backend_type_str=backend_name,
            initial_mapping_file_path=f"./experiment/{all_args.qasm_file_name}"
                                      f"/{all_args.backend_name}/initial_mapping_dir"
                                      f"/mapping_rank{ddp_rank}.txt",
            # randomness and buffer
            seed=env_seed,
            start_from_internal_prob=start_from_internal_prob,
            game_buffer_size=all_args.game_buffer_size,
            save_interval=all_args.game_buffer_save_interval,
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
        self.best_ln_fidelity = -10000  # ln fidelity is always smaller than 0 and the larger, the better
        self.fidelity_of_best_gate_count_circuit = -10000
        self.current_l2p_mapping = None

        # plan saving
        self.save_dir = save_dir
        self.save_threshold = save_threshold
        self.env_id = env_seed
        self.is_eval = is_eval
        self.start_time = time.time()

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

            # save the mapping
            self.current_l2p_mapping = next_state.logical2physical_mapping

        # put logging items in info
        cur_total_cost = -1
        cur_sum_ln_cx_fidelity = None
        final_mapping = None
        if done[0]:
            # get cost, fidelity and final mapping
            cur_total_cost = self.env.total_cost()
            cur_sum_ln_cx_fidelity = self.env.sum_ln_cx_fidelity()
            final_mapping = self.current_l2p_mapping

            # refresh the record
            _old_best_total_cost = self.best_total_cost
            self.best_total_cost = min(self.best_total_cost, cur_total_cost)
            if cur_total_cost < _old_best_total_cost:
                self.fidelity_of_best_gate_count_circuit = cur_sum_ln_cx_fidelity
            _old_best_ln_fidelity = self.best_ln_fidelity
            self.best_ln_fidelity = max(self.best_ln_fidelity, cur_sum_ln_cx_fidelity)

            # save the plan into wandb only if the plan refreshes the best plan found (and is good enough)
            if cur_total_cost <= self.save_threshold and (cur_total_cost < _old_best_total_cost
                                                          or cur_sum_ln_cx_fidelity > _old_best_ln_fidelity):
                time_stamp = int(time.time() - self.start_time)
                pre_prefix = "1g_" if cur_total_cost < _old_best_total_cost else "2f_"
                prefix = pre_prefix + ("eval" if self.is_eval else "collect")
                suffix = f"cost{cur_total_cost}_fidelity{cur_sum_ln_cx_fidelity}_{time_stamp}s_env{self.env_id}"
                execution_history_path = self.save_dir + "/" + prefix + "_execution_history_" + suffix + ".txt"
                single_qubit_gate_plan_path = self.save_dir + "/" + prefix + "_single_qubit_" + suffix + ".txt"
                output_qasm_path = self.save_dir + "/" + prefix + "_" + suffix + ".qasm"
                self.env.save_context(execution_history_file_path=execution_history_path,
                                      single_qubit_gate_execution_plan_file_path=single_qubit_gate_plan_path)
                self.env.generate_mapped_qasm(mapped_qasm_file_path=output_qasm_path, debug_mode=True)

        info = [{"individual_reward": reward[0][0], "best_total_cost": self.best_total_cost,
                 "cur_total_cost": cur_total_cost, "final_mapping": final_mapping,
                 "cur_fidelity": cur_sum_ln_cx_fidelity, "best_fidelity": self.best_ln_fidelity,
                 "fidelity_of_best_gate_count_circuit": self.fidelity_of_best_gate_count_circuit}]

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

    def render(self, mode='human'):
        raise NotImplementedError
