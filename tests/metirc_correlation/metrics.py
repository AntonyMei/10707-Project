from rollout_cost_metric import rollout_cost_metric
from sabre_cost_metric import sabre_cost_metric
from value_metric import value_metric


def save_mapping(filename, mapping):
    with open(filename, "w") as file:
        for i in range(len(mapping)):
            file.write(f"{mapping[i]} ")
        file.write("\n")


def evaluate_mapping(all_args, step, mapping, round_number, random_seed, cuda_idx=0):
    """
    Evaluate the mapping under different metrics
    :param all_args: arguments
    :param step: step number
    :param mapping: a dict or a list
    :param round_number: number of rounds
    :param random_seed: random seed
    :param cuda_idx: cuda index
    """
    # save mapping to file
    mapping_file_name = f"mapping_{hex(hash(str(mapping)))}.txt"
    save_mapping(filename="./mappings/" + mapping_file_name, mapping=mapping)

    # run evaluations
    min_sabre_cost, avg_sabre_cost, min_swap_count, avg_swap_count = sabre_cost_metric(
        qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
        mapping_file_name=mapping_file_name, round_number=round_number, random_seed=random_seed)
    eval_average_episode_rewards, eval_min_total_cost, eval_avg_total_cost = rollout_cost_metric(
        all_args=all_args, qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
        initial_mapping_file_path=mapping_file_name,
        model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
        round_number=2, seed=random_seed, deterministic=True, cuda_idx=cuda_idx)
    collect_average_episode_rewards, collect_min_total_cost, collect_avg_total_cost = rollout_cost_metric(
        all_args=all_args, qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
        initial_mapping_file_path=mapping_file_name,
        model_path="../../onpolicy/scripts/example_models/model_-3reward.pt",
        round_number=round_number, seed=random_seed, deterministic=False, cuda_idx=cuda_idx)
    value = value_metric(all_args=all_args, qasm_file_name=all_args.qasm_file_name,
                         device_name=all_args.backend_name, mapping_file_name=mapping_file_name,
                         model_path="../../onpolicy/scripts/example_models/model_-3reward.pt", cuda_idx=cuda_idx)

    # pack evaluation results
    results = {
        "step": step,
        "mapping": str(mapping),
        "min_sabre_cost": min_sabre_cost,
        "avg_sabre_cost": avg_sabre_cost,
        "min_swap_count": min_swap_count,
        "avg_swap_count": avg_swap_count,
        "eval_average_episode_rewards": float(eval_average_episode_rewards),
        "eval_min_total_cost": int(eval_min_total_cost),
        "eval_avg_total_cost": float(eval_avg_total_cost),
        "collect_average_episode_rewards": float(collect_average_episode_rewards),
        "collect_min_total_cost": int(collect_min_total_cost),
        "collect_avg_total_cost": float(collect_avg_total_cost),
        "value": value,
    }
    return results
