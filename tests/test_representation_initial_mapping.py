from quartz import PySimpleInitialEnv

import torch

from onpolicy.algorithms.quartz_ppo_dual.algorithm.representation_network import RepresentationNetwork
from onpolicy.algorithms.quartz_ppo_dual.algorithm.actor_critic import ValueNetwork, PolicyNetwork
from onpolicy.utils.quartz_utils import py_action_list_2_list


def main():
    # environment
    env1 = PySimpleInitialEnv(qasm_file_path="../onpolicy/scripts/qasm_files/barenco_tof_10_after_heavy.qasm",
                              backend_type_str="IBM_Q20_TOKYO")
    state1 = env1.get_state()
    graph_state1 = state1.circuit
    device_edge_list1 = state1.device_edges_list
    logical2physical1 = state1.logical2physical_mapping
    physical2logical1 = state1.physical2logical_mapping

    env2 = PySimpleInitialEnv(qasm_file_path="../onpolicy/scripts/qasm_files/barenco_tof_10_after_heavy.qasm",
                              backend_type_str="IBM_Q20_TOKYO")
    state2 = env2.get_state()
    graph_state2 = state2.circuit
    device_edge_list2 = state2.device_edges_list
    logical2physical2 = state2.logical2physical_mapping
    physical2logical2 = state2.physical2logical_mapping

    # representation
    network = RepresentationNetwork(reg_degree_types=8,
                                    reg_degree_embedding_dim=64,
                                    gate_is_input_embedding_dim=64,
                                    num_gnn_layers=6,
                                    reg_representation_dim=128,
                                    gate_representation_dim=128,
                                    device=torch.device("cuda"))
    reg_representation_batch, reg_count = network(circuit_batch=[graph_state1, graph_state2],
                                                  device_edge_list_batch=[device_edge_list1, device_edge_list2],
                                                  physical2logical_mapping_batch=[physical2logical1, physical2logical2],
                                                  logical2physical_mapping_batch=[logical2physical1, logical2physical2])

    # value
    value_net = ValueNetwork(register_embedding_dimension=256, device=torch.device("cuda"))
    value = value_net(reg_representation_batch, reg_count)
    print(f"{value=}")

    # policy
    action_space = [py_action_list_2_list(env1.get_action_space()) + [(0, 0)],
                    py_action_list_2_list(env2.get_action_space()) + [(0, 0)]]
    policy_net = PolicyNetwork(register_embedding_dimension=256, device=torch.device("cuda"), allow_nop=True)
    policy = policy_net(reg_representation_batch, reg_count, action_space)
    print(f"{policy=}")


if __name__ == '__main__':
    main()
