import sys

from onpolicy.config import get_config
from onpolicy.runner.quartz.initial_mapping_search import random_search_round, random_search, save_mapping


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--world_size", type=int, required=True, help="DDP world size")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # parse argument
    parser = get_config()
    all_args = parse_args(args, parser)

    # round api test
    print("Round Test:")
    result = random_search_round(all_args=all_args, ddp_rank=0, round_seed=0,
                                 model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
                                 mapping_file_path="./mapping.txt")
    print(result)
    print()

    # full api test
    full_res = random_search(all_args=all_args, ddp_rank=0, episode=0,
                             model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
                             mapping_file_path="./mapping.txt")
    print(full_res)
    save_mapping("./random.txt", full_res)


if __name__ == '__main__':
    main(sys.argv[1:])
