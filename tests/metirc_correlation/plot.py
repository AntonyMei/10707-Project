import matplotlib.pyplot as plt
import json


def main():
    # choose plot type "bfs" or "random"
    plot_type = "random"

    # load data from json file
    if plot_type == "bfs":
        with open("./bfs_result/bfs_result_1700.json", "r") as f:
            result_dict = json.load(f)
    elif plot_type == "random":
        with open("./random_result/random_result_1_4700.json", "r") as f1:
            result_dict = json.load(f1)
        with open("./random_result/random_result_2_4500.json", "r") as f2:
            result_dict.update(json.load(f2))
        with open("./random_result/random_result_3_7000.json", "r") as f3:
            result_dict.update(json.load(f3))
    else:
        print("Error: plot type not found")
        return
    print(f"({plot_type}) Result dict size: {len(result_dict)}")

    # unpack data from results dict
    step_list = []
    mapping_list = []
    min_sabre_cost_list = []
    avg_sabre_cost_list = []
    min_swap_count_list = []
    avg_swap_count_list = []
    eval_average_episode_rewards_list = []
    eval_min_total_cost_list = []
    eval_avg_total_cost_list = []
    collect_average_episode_rewards_list = []
    collect_min_total_cost_list = []
    collect_avg_total_cost_list = []
    value_list = []
    for mapping in result_dict:
        result = result_dict[mapping]
        step_list.append(result["step"])
        mapping_list.append(result["mapping"])
        min_sabre_cost_list.append(result["min_sabre_cost"])
        avg_sabre_cost_list.append(result["avg_sabre_cost"])
        min_swap_count_list.append(result["min_swap_count"])
        avg_swap_count_list.append(result["avg_swap_count"])
        eval_average_episode_rewards_list.append(result["eval_average_episode_rewards"])
        eval_min_total_cost_list.append(result["eval_min_total_cost"])
        eval_avg_total_cost_list.append(result["eval_avg_total_cost"])
        collect_average_episode_rewards_list.append(result["collect_average_episode_rewards"])
        collect_min_total_cost_list.append(result["collect_min_total_cost"])
        collect_avg_total_cost_list.append(result["collect_avg_total_cost"])
        value_list.append(result["value"])

    # clip each entry in data as 1000
    min_sabre_cost_list = [min(1000, cost) for cost in min_sabre_cost_list]
    eval_min_total_cost_list = [min(1000, cost) for cost in eval_min_total_cost_list]
    eval_avg_total_cost_list = [min(1000, cost) for cost in eval_avg_total_cost_list]
    collect_min_total_cost_list = [min(1000, cost) for cost in collect_min_total_cost_list]
    collect_avg_total_cost_list = [min(2000, cost) for cost in collect_avg_total_cost_list]

    # set some parameters
    plt.rcParams.update({'font.size': 24})      # set font size
    plt.rcParams['figure.subplot.left'] = 0.15  # set left margin
    plt.rcParams['axes.labelpad'] = 10          # set label padding
    plt.rcParams['axes.titlepad'] = 20          # set title padding

    # plot a scatter plot of value and collect_min_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(value_list, collect_min_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Value")
    ax.set_ylabel("Collect Min Total Cost")
    ax.set_title("Value vs Collect Min Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/value_vs_collect_min_total_cost.png")
    plt.show()

    # plot a scatter plot of value and collect_avg_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(value_list, collect_avg_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Value")
    ax.set_ylabel("Collect Avg Total Cost")
    ax.set_title("Value vs Collect Avg Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/value_vs_collect_avg_total_cost.png")
    plt.show()

    # plot a scatter plot of value and eval_min_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(value_list, eval_min_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Value")
    ax.set_ylabel("Eval Min Total Cost")
    ax.set_title("Value vs Eval Min Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/value_vs_eval_min_total_cost.png")
    plt.show()
    
    # plot a scatter plot of min_sabre_cost_list and collect_min_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(min_sabre_cost_list, collect_min_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Min Sabre Cost")
    ax.set_ylabel("Collect Min Total Cost")
    ax.set_title("Min Sabre Cost vs Collect Min Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/min_sabre_cost_vs_collect_min_total_cost.png")
    plt.show()
    
    # plot a scatter plot of min_sabre_cost_list and collect_avg_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(min_sabre_cost_list, collect_avg_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Min Sabre Cost")
    ax.set_ylabel("Collect Avg Total Cost")
    ax.set_title("Min Sabre Cost vs Collect Avg Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/min_sabre_cost_vs_collect_avg_total_cost.png")
    plt.show()
    
    # plot a scatter plot of min_sabre_cost_list and eval_min_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(min_sabre_cost_list, eval_min_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Min Sabre Cost")
    ax.set_ylabel("Eval Min Total Cost")
    ax.set_title("Min Sabre Cost vs Eval Min Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/min_sabre_cost_vs_eval_min_total_cost.png")
    plt.show()
    
    # plot a scatter plot of min_sabre_cost_list and eval_avg_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(min_sabre_cost_list, eval_avg_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Min Sabre Cost")
    ax.set_ylabel("Eval Avg Total Cost")
    ax.set_title("Min Sabre Cost vs Eval Avg Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/min_sabre_cost_vs_eval_avg_total_cost.png")
    plt.show()
    
    # plot a scatter plot of eval_min_total_cost_list and collect_min_total_cost_list, with step as category
    fig, ax = plt.subplots(figsize=(20, 10), dpi=100)
    ax.scatter(eval_min_total_cost_list, collect_min_total_cost_list, c=step_list, cmap="viridis")
    ax.set_xlabel("Eval Min Total Cost")
    ax.set_ylabel("Collect Min Total Cost")
    ax.set_title("Eval Min Total Cost vs Collect Min Total Cost")
    cbar = fig.colorbar(ax.collections[0])
    cbar.set_label("Step")
    plt.savefig(f"./{plot_type}_figures/eval_min_total_cost_vs_collect_min_total_cost.png")
    plt.show()


if __name__ == '__main__':
    main()
