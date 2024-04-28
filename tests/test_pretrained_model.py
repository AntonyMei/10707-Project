from quartz import PySimpleHybridEnv, PySimplePhysicalEnv

import torch

from onpolicy.algorithms.quartz_ppo_dual.algorithm.representation_network import RepresentationNetwork
from onpolicy.algorithms.quartz_ppo_dual.algorithm.actor_critic import ValueNetwork, PolicyNetwork
from onpolicy.algorithms.utils.util import DummyObjWrapper
from onpolicy.utils.quartz_utils import py_action_list_2_list


def main():
    # model path
    model_path = "../onpolicy/scripts/pretrained_models/IBM_Q65_Hummingbird_model.pt"

    # environment
    env1 = PySimpleHybridEnv(
        qasm_file_path="../onpolicy/scripts/qasm_files/adder_8_after_heavy.qasm",
        backend_type_str="IBM_Q65_HUMMINGBIRD", initial_mapping_file_path="./mapping65.txt",
        seed=0, start_from_internal_prob=0.2,
        initial_phase_len=2, allow_nop_in_initial=True, initial_phase_reward=0,
        game_buffer_size=200, save_interval=5)
    state1 = env1.get_state()
    graph_state1 = state1.circuit
    device_edge_list1 = state1.device_edges_list
    logical2physical1 = state1.logical2physical_mapping
    physical2logical1 = state1.physical2logical_mapping
    is_initial1 = state1.is_initial_phase

    env2 = PySimpleHybridEnv(
        qasm_file_path="../onpolicy/scripts/qasm_files/adder_8_after_heavy_reversed.qasm",
        backend_type_str="IBM_Q65_HUMMINGBIRD", initial_mapping_file_path="./mapping65.txt",
        seed=1, start_from_internal_prob=0.2,
        initial_phase_len=2, allow_nop_in_initial=True, initial_phase_reward=0,
        game_buffer_size=200, save_interval=5)
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
    # load model
    # we need to exclude value & policy network parameters from the pretrained model
    model_file_name = model_path
    model_state_dict = torch.load(model_file_name)
    representation_network_state_dict = {}
    for k, v in model_state_dict.items():
        # ignore value and policy network parameters
        if k.startswith("policy") or k.startswith("value"):
            continue

        # rename representation network parameters
        if k.startswith("representation_network."):
            name = k[23+7:]  # remove `representation_network.`
        else:
            name = k
        representation_network_state_dict[name] = v
    network.load_state_dict(representation_network_state_dict)
    print(f"Info: --pretrain_mode=representation, Representation network is successfully "
          f"restored from {model_file_name}!")

    # forward
    reg_representation_batch, reg_count = network.forward(
        circuit_batch=DummyObjWrapper([graph_state1, graph_state2]),
        device_edge_list_batch=DummyObjWrapper([device_edge_list1, device_edge_list2]),
        physical2logical_mapping_batch=DummyObjWrapper([physical2logical1, physical2logical2]),
        logical2physical_mapping_batch=DummyObjWrapper([logical2physical1, logical2physical2]))

    # value
    is_initial_batch = [is_initial1, is_initial2]
    print(is_initial_batch)
    value_net = ValueNetwork(register_embedding_dimension=256, device=torch.device("cuda"))
    value = value_net.forward(DummyObjWrapper(reg_representation_batch),
                              DummyObjWrapper(reg_count),
                              DummyObjWrapper(is_initial_batch))
    print(f"{value=}")
    loss = value.sum()
    loss.backward()
    print(loss.grad, "!")

    # policy
    action_space = [py_action_list_2_list(env1.get_action_space()),
                    py_action_list_2_list(env2.get_action_space())]
    policy_net = PolicyNetwork(register_embedding_dimension=256, device=torch.device("cuda"), allow_nop=True)
    policy = policy_net.forward(DummyObjWrapper(reg_representation_batch),
                                DummyObjWrapper(reg_count),
                                DummyObjWrapper(action_space))
    print(f"{policy=}")


if __name__ == '__main__':
    for _ in range(1):
        main()
        print()
