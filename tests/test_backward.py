from quartz import PySimpleHybridEnv, PySimplePhysicalEnv

import time
import torch

from onpolicy.algorithms.quartz_ppo_dual.algorithm.representation_network import RepresentationNetwork
from onpolicy.algorithms.quartz_ppo_dual.algorithm.actor_critic import ValueNetwork, PolicyNetwork
from onpolicy.utils.quartz_utils import py_action_list_2_list
from onpolicy.algorithms.utils.util import DummyObjWrapper


def main():
    # environment
    env1 = PySimpleHybridEnv(
        qasm_file_path="../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
        backend_type_str="IBM_Q27_FALCON", initial_mapping_file_path="./mapping.txt",
        seed=0, start_from_internal_prob=0,
        initial_phase_len=2, allow_nop_in_initial=True, initial_phase_reward=-0.3)
    state1 = env1.get_state()
    graph_state1 = state1.circuit
    device_edge_list1 = state1.device_edges_list
    logical2physical1 = state1.logical2physical_mapping
    physical2logical1 = state1.physical2logical_mapping
    is_initial1 = state1.is_initial_phase

    env2 = PySimplePhysicalEnv(qasm_file_path="../onpolicy/scripts/qasm_files/csla_mux_3_after_heavy.qasm",
                               backend_type_str="IBM_Q27_FALCON", seed=2, start_from_internal_prob=0.5,
                               initial_mapping_file_path="./mapping.txt")
    state2 = env2.get_state()
    graph_state2 = state2.circuit
    device_edge_list2 = state2.device_edges_list
    logical2physical2 = state2.logical2physical_mapping
    physical2logical2 = state2.physical2logical_mapping
    is_initial2 = state2.is_initial_phase

    # representation
    network = RepresentationNetwork(reg_degree_types=8,
                                    reg_degree_embedding_dim=64,
                                    gate_is_input_embedding_dim=64,
                                    num_gnn_layers=6,
                                    reg_representation_dim=128,
                                    gate_representation_dim=128,
                                    device=torch.device("cuda"))
    reg_representation_batch, reg_count = network(circuit_batch=DummyObjWrapper([graph_state1, graph_state2] * 1200),
                                                  device_edge_list_batch=DummyObjWrapper([device_edge_list1, device_edge_list2] * 1200),
                                                  physical2logical_mapping_batch=DummyObjWrapper([physical2logical1, physical2logical2] * 1200),
                                                  logical2physical_mapping_batch=DummyObjWrapper([logical2physical1, logical2physical2] * 1200))

    # value
    is_initial_batch = [is_initial1, is_initial2]
    print(is_initial_batch)
    value_net = ValueNetwork(register_embedding_dimension=256, device=torch.device("cuda"))
    value = value_net(DummyObjWrapper(reg_representation_batch),
                      DummyObjWrapper(reg_count),
                      DummyObjWrapper(is_initial_batch))
    # print(f"{value=}")
    value_bp = torch.sum(value)
    # value_backward_start = time.time()
    # value_bp.backward()
    # value_backward_end = time.time()
    # print(f"{value_backward_end - value_backward_start=}")

    # policy
    action_space = [py_action_list_2_list(env1.get_action_space()),
                    py_action_list_2_list(env2.get_action_space())]
    policy_net = PolicyNetwork(register_embedding_dimension=256, device=torch.device("cuda"), allow_nop=True)
    policy = policy_net(DummyObjWrapper(reg_representation_batch),
                        DummyObjWrapper(reg_count),
                        DummyObjWrapper(action_space))
    # print(f"{policy=}")
    policy_bp = sum([torch.sum(ele) for ele in policy])
    # policy_backward_start = time.time()
    # policy_bp.backward()
    # policy_backward_end = time.time()
    # print(f"{policy_backward_end - policy_backward_start=}")

    # total
    total_loss = value_bp + policy_bp
    total_start = time.time()
    total_loss.backward()
    total_end = time.time()
    print(f"{total_end - total_start=}")


if __name__ == '__main__':
    for _ in range(1):
        main()
        print()
