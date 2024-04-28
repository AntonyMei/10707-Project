# this must be the first import
from quartz import PySimpleInitialEnv, PyAction

import time
import random
import gym
import numpy as np

from gym import spaces
from onpolicy.utils.quartz_utils import flatten_observation, restore_action
from onpolicy.utils.quartz_utils import build_qiskit_circuit, get_qiskit_sabre_cost
from onpolicy.envs.quartz_initial_mapping.topologies import parse_coupling_graph_name


class SimpleInitialEnv(gym.Env):
    def __init__(self, qasm_file_name, backend_name, all_args, max_obs_length=2000):
        self.all_args = all_args

        # prepare underlying c++ env and qiskit env
        self.env = PySimpleInitialEnv(qasm_file_path=qasm_file_name, backend_type_str=backend_name)
        self.coupling_map, num_regs = parse_coupling_graph_name(name=backend_name)
        self.qiskit_circuit = build_qiskit_circuit(qasm_file_path=qasm_file_name, num_regs=num_regs)

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

        # reward and data collecting statistics
        self.cur_qiskit_cost = get_qiskit_sabre_cost(state=state, circuit=self.qiskit_circuit,
                                                     coupling_map=self.coupling_map)
        self.best_total_cost = self.cur_qiskit_cost
        self.global_best_total_cost = self.cur_qiskit_cost  # this won't be cleared after one episode
        self.cur_step = 0

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action):
        # action here is a number that needs decoding
        qubit_idx_0, qubit_idx_1 = restore_action(action)

        # apply action and get reward & check finished
        if qubit_idx_0 == 0 and qubit_idx_1 == 0:
            # NOP will end the game
            reward = [[0]]
            done = [True]
            new_state = None
            new_qiskit_cost = self.cur_qiskit_cost
        else:
            self.env.step_with_id(qubit_idx_0, qubit_idx_1)
            new_state = self.env.get_state()
            new_qiskit_cost = get_qiskit_sabre_cost(state=new_state, circuit=self.qiskit_circuit,
                                                    coupling_map=self.coupling_map)
            reward = [[self.all_args.initial_env_penalty_threshold - new_qiskit_cost]]
            done = [self.cur_step == self.all_args.episode_length - 1]

            # save new initial mapping to disk if it is a good mapping
            if new_qiskit_cost <= self.all_args.initial_env_save_threshold:
                # prepare logical -> physical mapping
                logical2physical_mapping = new_state.logical2physical_mapping
                logical2physical_list = []
                for k in range(len(logical2physical_mapping)):
                    logical2physical_list.append(logical2physical_mapping[k])

                # write into file
                folder_name = f"./experiment/{self.all_args.qasm_file_name}/{self.all_args.backend_name}" \
                              f"/initial_mapping_dir/"
                file_name = f"{new_qiskit_cost}_{int(time.time() * 1000)}_{random.randint(0, 10000)}.initial"
                with open(folder_name + file_name, "w") as file_handle:
                    file_handle.write(f"{logical2physical_list}")

        # update env parameters
        self.cur_qiskit_cost = new_qiskit_cost
        self.best_total_cost = min(self.best_total_cost, new_qiskit_cost)
        self.global_best_total_cost = min(self.global_best_total_cost, new_qiskit_cost)
        self.cur_step += 1

        # get next obs
        if done[0]:
            next_obs = None
        else:
            next_state = new_state
            action_space = self.env.get_action_space()
            nop_action = PyAction(type_str="PhysicalFull", qubit_idx_0=0, qubit_idx_1=0)
            action_space.append(nop_action)
            flattened_obs = flatten_observation(graph_state=next_state.circuit,
                                                device_edge_list=next_state.device_edges_list,
                                                logical2physical=next_state.logical2physical_mapping,
                                                physical2logical=next_state.physical2logical_mapping,
                                                action_space=action_space,
                                                is_initial_phase=False,
                                                target_length=self._max_observation_length)
            next_obs = [flattened_obs]

        # put logging items in info
        info = [{"individual_reward": reward[0][0],
                 "best_total_cost": self.global_best_total_cost,    # used by collect
                 "cur_total_cost": self.best_total_cost}]           # used by evaluation

        return next_obs, reward, done, info

    def reset(self):
        # reset environment
        self.env.reset()
        self.cur_step = 0

        # return new observation
        next_state = self.env.get_state()
        action_space = self.env.get_action_space()
        nop_action = PyAction(type_str="PhysicalFull", qubit_idx_0=0, qubit_idx_1=0)
        action_space.append(nop_action)
        flattened_obs = flatten_observation(graph_state=next_state.circuit,
                                            device_edge_list=next_state.device_edges_list,
                                            logical2physical=next_state.logical2physical_mapping,
                                            physical2logical=next_state.physical2logical_mapping,
                                            action_space=action_space,
                                            is_initial_phase=False,
                                            target_length=self._max_observation_length)
        next_obs = [flattened_obs]

        # reset environment
        self.cur_qiskit_cost = get_qiskit_sabre_cost(state=next_state, circuit=self.qiskit_circuit,
                                                     coupling_map=self.coupling_map)
        self.best_total_cost = self.cur_qiskit_cost

        return next_obs

    def render(self, mode='human'):
        raise NotImplementedError
