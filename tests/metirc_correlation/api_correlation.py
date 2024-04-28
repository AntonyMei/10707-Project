import sys

from onpolicy.config import get_config
from onpolicy.runner.quartz.initial_mapping_search import random_search_round, random_search
from sabre_cost_metric import sabre_cost_metric
from rollout_cost_metric import rollout_cost_metric


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--world_size", type=int, required=True, help="DDP world size")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def save_mapping(filename, mapping):
    with open(filename, "w") as file:
        for i in range(len(mapping)):
            file.write(f"{mapping[i]} ")
        file.write("\n")


def main(args):
    # parse argument
    parser = get_config()
    all_args = parse_args(args, parser)

    # full api test
    full_res = random_search(all_args=all_args, ddp_rank=0, episode=0,
                             model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
                             mapping_file_path="./mapping.txt")

    # test the results under different metrics
    for value in full_res:
        # get mapping
        mapping = full_res[value]
        mapping_file_name = f"mapping_{hash(str(mapping))}.txt"
        save_mapping(filename="./mappings/" + mapping_file_name, mapping=mapping)

        # run different metrics
        min_sabre_cost, avg_sabre_cost, min_swap_count, avg_swap_count = sabre_cost_metric(
            qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
            mapping_file_name=mapping_file_name, round_number=16, random_seed=0)
        eval_average_episode_rewards, eval_min_total_cost, eval_avg_total_cost = rollout_cost_metric(
            all_args=all_args, qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
            initial_mapping_file_path=mapping_file_name,
            model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
            round_number=2, seed=0, deterministic=True)
        collect_average_episode_rewards, collect_min_total_cost, collect_avg_total_cost = rollout_cost_metric(
            all_args=all_args, qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
            initial_mapping_file_path=mapping_file_name,
            model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
            round_number=32, seed=0, deterministic=False)

        # print results
        print(f"mapping: {mapping}, value={value}")
        print("Sabre:")
        print(f"sabre cost: min={min_sabre_cost}, avg={avg_sabre_cost}")
        print(f"sabre swap count: min={min_swap_count}, avg={avg_swap_count}")
        print("Eval rollout:")
        print(f"eval average episode rewards: {eval_average_episode_rewards}")
        print(f"eval min total cost: {eval_min_total_cost}")
        print(f"eval avg total cost: {eval_avg_total_cost}")
        print("Collect rollout:")
        print(f"collect average episode rewards: {collect_average_episode_rewards}")
        print(f"collect min total cost: {collect_min_total_cost}")
        print(f"collect avg total cost: {collect_avg_total_cost}")
        print("")


if __name__ == '__main__':
    main(sys.argv[1:])
