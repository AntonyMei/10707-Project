import math
import random
import sys
import time
import os
import json


from quartz import PySimpleSearchEnv

from onpolicy.algorithms.quartz_ppo_dual.algorithm.quartz_ppo_network import QuartzPPONetwork
from onpolicy.config import get_config
from metrics import evaluate_mapping

import torch


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--world_size", type=int, required=True, help="DDP world size")
    parser.add_argument("--cuda_idx", type=int, required=True, help="cuda index")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def save_mapping(filename, mapping):
    with open(filename, "a") as file:
        for i in range(len(mapping)):
            file.write(f"{mapping[i]} ")
        file.write("\n")


def main(args):
    # parse argument
    parser = get_config()
    all_args = parse_args(args, parser)

    # some parameters
    round_number = 32
    seed = all_args.cuda_idx + all_args.seed

    # initialize env
    env = PySimpleSearchEnv(qasm_file_path="../../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
                            backend_type_str="IBM_Q27_FALCON", seed=0, start_from_internal_prob=0,
                            initial_mapping_file_path="./mapping.txt")

    # initialize network
    network = QuartzPPONetwork(reg_degree_types=all_args.reg_degree_types,
                               reg_degree_embedding_dim=all_args.reg_degree_embedding_dim,
                               gate_is_input_embedding_dim=all_args.gate_is_input_embedding_dim,
                               num_gnn_layers=all_args.num_gnn_layers,
                               reg_representation_dim=all_args.reg_representation_dim,
                               gate_representation_dim=all_args.gate_representation_dim,
                               device=torch.device(f"cuda:{all_args.cuda_idx}"),
                               rank=None, allow_nop=False)
    raw_state_dict = torch.load("../../onpolicy/scripts/example_models/model_-3reward.pt")
    state_dict = {'.'.join([k.split('.')[0]] + k.split('.')[2:]): v.cpu() for k, v in raw_state_dict.items()}
    network.load_state_dict(state_dict=state_dict)

    # random search
    best_value = -10000
    step = 0
    result_dict = {}
    index_dict = {}
    index = 0
    transition_count = 0

    # final metric result
    searched_count = 0
    start_time = time.time()
    final_metric_dict = {}

    while step < 100000:
        # append current state to result dict
        cur_state = env.get_state()
        value = network.value_forward(circuit_batch=[cur_state.circuit],
                                      device_edge_list_batch=[cur_state.device_edges_list],
                                      physical2logical_mapping_batch=[cur_state.physical2logical_mapping],
                                      logical2physical_mapping_batch=[cur_state.logical2physical_mapping],
                                      is_initial_phase_batch=[cur_state.is_initial_phase])
        value = float(value) + 3
        if value not in result_dict:
            result_dict[value] = [cur_state]
        else:
            result_dict[value].append(cur_state)

        # save final metric result to dict
        cur_mapping = cur_state.logical2physical_mapping
        if str(cur_mapping) not in final_metric_dict:
            metrics = evaluate_mapping(all_args=all_args, step=step, mapping=cur_mapping,
                                       round_number=round_number, random_seed=seed, cuda_idx=all_args.cuda_idx)
            searched_count += 1
            final_metric_dict[str(cur_mapping)] = metrics

        # save into index dict
        str_cur_mapping = f"{cur_state.logical2physical_mapping}"
        if str_cur_mapping not in index_dict:
            index_dict[str_cur_mapping] = index
            index += 1

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
        new_value = float(new_value) + 3
        # print(value, new_value)

        # random search
        if value < new_value:
            env = env_copy
            transition_count += 1
            threshold = 1
            move_to_new_state = True
        else:
            threshold = math.exp(-20 * (value - new_value))
            # print(threshold)
            random_number = random.random()
            move_to_new_state = False
            if random_number < threshold:
                env = env_copy
                transition_count += 1
                move_to_new_state = True

        # save new mapping into index dict
        str_new_mapping = f"{new_state.logical2physical_mapping}"
        if str_new_mapping not in index_dict:
            index_dict[str_new_mapping] = index
            index += 1
            new_state_seen = False
        else:
            new_state_seen = True

        # log each transition
        assert not str_new_mapping == str_cur_mapping
        # print(f"In step {step}, "
        #       f"Current state index: {index_dict[str_cur_mapping]} ({value}).\n"
        #       f"Proposed state is: {index_dict[str_new_mapping]} ({new_value}, seen={new_state_seen}).\n"
        #       f"Transition Probability = {threshold}, transit to new state = {move_to_new_state}, "
        #       f"transition count = {transition_count}.\n")

        # log
        step += 1
        if new_value > best_value:
            best_value = new_value
            print(f"Step {step}: new best value found: {best_value}.")
            save_mapping("search_result.txt", new_state.logical2physical_mapping)
        # print()

        if searched_count % 100 == 0:
            print(f"searched {searched_count} mappings, time elapsed: {time.time() - start_time:.2f}s.")
            os.makedirs("./random_result", exist_ok=True)
            with open(f"./random_result/random_result_{all_args.cuda_idx}_{searched_count}.json", "w") as f:
                json.dump(final_metric_dict, f)


if __name__ == '__main__':
    main(sys.argv[1:])
