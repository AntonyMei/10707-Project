import math
import random
import sys
import time


from quartz import PySimpleSearchEnv

from onpolicy.algorithms.quartz_ppo_dual.algorithm.quartz_ppo_network import QuartzPPONetwork
from onpolicy.config import get_config

import torch


def value_metric(all_args, qasm_file_name, device_name, mapping_file_name, model_path, cuda_idx=0):
    # initialize env
    env = PySimpleSearchEnv(qasm_file_path="../search/qasm_files/" + qasm_file_name,
                            backend_type_str=device_name, seed=0, start_from_internal_prob=0,
                            initial_mapping_file_path="./mappings/" + mapping_file_name)

    # initialize network
    network = QuartzPPONetwork(reg_degree_types=all_args.reg_degree_types,
                               reg_degree_embedding_dim=all_args.reg_degree_embedding_dim,
                               gate_is_input_embedding_dim=all_args.gate_is_input_embedding_dim,
                               num_gnn_layers=all_args.num_gnn_layers,
                               reg_representation_dim=all_args.reg_representation_dim,
                               gate_representation_dim=all_args.gate_representation_dim,
                               device=torch.device(f"cuda:{cuda_idx}"),
                               rank=None, allow_nop=False)
    raw_state_dict = torch.load(model_path)
    state_dict = {'.'.join([k.split('.')[0]] + k.split('.')[2:]): v.cpu() for k, v in raw_state_dict.items()}
    network.load_state_dict(state_dict=state_dict)

    # get value
    cur_state = env.get_state()
    value = network.value_forward(circuit_batch=[cur_state.circuit],
                                  device_edge_list_batch=[cur_state.device_edges_list],
                                  physical2logical_mapping_batch=[cur_state.physical2logical_mapping],
                                  logical2physical_mapping_batch=[cur_state.logical2physical_mapping],
                                  is_initial_phase_batch=[cur_state.is_initial_phase])
    value = float(value) + 3
    return value
