import os
import sys
import argparse
import json
import time

from quartz import PySimpleSearchEnv
from metrics import evaluate_mapping
from onpolicy.config import get_config


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--world_size", type=int, required=True, help="DDP world size")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def bfs_correlation(args):
    # some global parameters
    round_number = 32
    random_seed = 0

    # parse argument
    parser = get_config()
    all_args = parse_args(args, parser)

    # prepare environment
    query_queue = []
    for idx in range(8):
        env = PySimpleSearchEnv(qasm_file_path="../../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
                                backend_type_str="IBM_Q27_FALCON", seed=0, start_from_internal_prob=0,
                                initial_mapping_file_path=f"./mappings/mapping{idx}")
        query_queue.append((env, 0))

    # prepare result dict
    # result_dict: {mapping str: metrics_dict}
    result_dict = {}

    # start bfs search
    searched_count = 0
    start_time = time.time()
    while len(query_queue) > 0:
        # pop from queue
        env, step = query_queue.pop(0)

        # get current mapping
        current_mapping = env.get_state().logical2physical_mapping
        current_mapping_str = str(current_mapping)

        # check if current mapping is in result dict
        if current_mapping_str in result_dict:
            # if yes, check if current step is smaller
            if step < result_dict[current_mapping_str]["step"]:
                # if yes, update step
                result_dict[current_mapping_str]["step"] = step
        else:
            # if no, evaluate mapping
            metrics = evaluate_mapping(all_args=all_args, step=step, mapping=current_mapping,
                                       round_number=round_number, random_seed=random_seed)
            searched_count += 1

            # add to result dict
            result_dict[current_mapping_str] = metrics

            # add neighbors to queue
            for action in env.get_action_space():
                env_copy = env.copy()
                env_copy.step(action)
                query_queue.append((env_copy, step + 1))

        # save result dict
        if searched_count % 100 == 0:
            print(f"searched {searched_count} mappings, time elapsed: {time.time() - start_time:.2f}s.")
            os.makedirs("./bfs_result", exist_ok=True)
            with open(f"./bfs_result/bfs_result_{searched_count}.json", "w") as f:
                json.dump(result_dict, f)


if __name__ == "__main__":
    bfs_correlation(sys.argv[1:])
