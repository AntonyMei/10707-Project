# this should always be the first
from quartz import PySimpleSearchEnv

# other imports
import math
import random
import sys
import torch

from onpolicy.algorithms.quartz_ppo_dual.algorithm.quartz_ppo_network import QuartzPPONetwork
from onpolicy.config import get_config
from tqdm import tqdm


def save_mapping(filename, mapping_dict):
    for value in mapping_dict:
        mapping = mapping_dict[value]
        with open(filename, "a") as file:
            for i in range(len(mapping)):
                file.write(f"{mapping[i]} ")
            file.write("\n")


def random_search_round(all_args, ddp_rank, round_seed, mapping_file_path, model_path):
    # initialize env
    env = PySimpleSearchEnv(qasm_file_path="qasm_files/" + all_args.qasm_file_name,
                            backend_type_str=all_args.backend_name,
                            seed=round_seed,
                            start_from_internal_prob=0,
                            initial_mapping_file_path=mapping_file_path)
    random.seed(round_seed)

    # initialize network
    network = QuartzPPONetwork(reg_degree_types=all_args.reg_degree_types,
                               reg_degree_embedding_dim=all_args.reg_degree_embedding_dim,
                               gate_is_input_embedding_dim=all_args.gate_is_input_embedding_dim,
                               num_gnn_layers=all_args.num_gnn_layers,
                               reg_representation_dim=all_args.reg_representation_dim,
                               gate_representation_dim=all_args.gate_representation_dim,
                               device=torch.device(f"cuda:{ddp_rank}"),
                               rank=None, allow_nop=False)
    raw_state_dict = torch.load(model_path)
    state_dict = {'.'.join([k.split('.')[0]] + k.split('.')[2:]): v.cpu() for k, v in raw_state_dict.items()}
    network.load_state_dict(state_dict=state_dict)

    # random search
    step = 0
    raw_result_dict = {}
    while step < all_args.round_budget:
        # append current state to result dict
        cur_state = env.get_state()
        value = network.value_forward(circuit_batch=[cur_state.circuit],
                                      device_edge_list_batch=[cur_state.device_edges_list],
                                      physical2logical_mapping_batch=[cur_state.physical2logical_mapping],
                                      logical2physical_mapping_batch=[cur_state.logical2physical_mapping],
                                      is_initial_phase_batch=[cur_state.is_initial_phase])
        value = float(value)
        raw_result_dict[value] = cur_state.logical2physical_mapping

        # make a random move
        env_copy = env.copy()
        action_space = env_copy.get_action_space()
        selected_action_id = random.randint(0, len(action_space) - 1)
        env_copy.step(action_space[selected_action_id])
        new_state = env_copy.get_state()
        new_value = network.value_forward(circuit_batch=[new_state.circuit],
                                          device_edge_list_batch=[new_state.device_edges_list],
                                          physical2logical_mapping_batch=[new_state.physical2logical_mapping],
                                          logical2physical_mapping_batch=[new_state.logical2physical_mapping],
                                          is_initial_phase_batch=[new_state.is_initial_phase])
        new_value = float(new_value)
        # assert value > 0, f"Error: value={value}!"
        # assert new_value > 0, f"Error: new value={new_value}!"

        # random transition
        step += 1
        if value < new_value:
            env = env_copy
        else:
            threshold = math.exp(-all_args.random_search_lambda * (value - new_value))
            random_number = random.random()
            if random_number < threshold:
                env = env_copy

    # return the final state as result
    final_state = env.get_state()
    final_mapping = final_state.logical2physical_mapping
    final_value = network.value_forward(circuit_batch=[final_state.circuit],
                                        device_edge_list_batch=[final_state.device_edges_list],
                                        physical2logical_mapping_batch=[final_state.physical2logical_mapping],
                                        logical2physical_mapping_batch=[final_state.logical2physical_mapping],
                                        is_initial_phase_batch=[final_state.is_initial_phase])
    return final_mapping, final_value


def random_search(all_args, ddp_rank, episode, mapping_file_path, model_path):
    # collect new initial mappings from rounds of search
    raw_results_dict = {}
    process_bar = tqdm(range(all_args.search_rounds)) if ddp_rank == 0 else range(all_args.search_rounds)
    for round_id in process_bar:
        round_seed = all_args.seed + episode + round_id * 10000 + ddp_rank * 1000000
        round_result_mapping, round_result_value = random_search_round(all_args=all_args,
                                                                       ddp_rank=ddp_rank,
                                                                       round_seed=round_seed,
                                                                       mapping_file_path=mapping_file_path,
                                                                       model_path=model_path)
        raw_results_dict[round_result_value] = round_result_mapping

    # select best mappings
    best_values = sorted(raw_results_dict, reverse=True)[:all_args.save_count]
    final_result_dict = {}
    for value in best_values:
        final_result_dict[value] = raw_results_dict[value]
    return final_result_dict
