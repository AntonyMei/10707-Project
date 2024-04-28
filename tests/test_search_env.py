import argparse
from quartz import PySimpleSearchEnv, PyAction


def test_same(env_state, env_copy_state):
    print(f"Pass copy test 1.1: {env_state.circuit.number_of_nodes == env_copy_state.circuit.number_of_nodes}")
    print(f"Pass copy test 1.2: {env_state.circuit.node_id == env_copy_state.circuit.node_id}")
    print(f"Pass copy test 1.3: {env_state.circuit.is_input == env_copy_state.circuit.is_input}")
    print(f"Pass copy test 1.4: {env_state.circuit.input_logical_idx == env_copy_state.circuit.input_logical_idx}")
    print(f"Pass copy test 1.5: {env_state.circuit.input_physical_idx == env_copy_state.circuit.input_physical_idx}")
    print(f"Pass copy test 1.6: {env_state.circuit.node_type == env_copy_state.circuit.node_type}")
    print(f"Pass copy test 1.7: {env_state.circuit.number_of_edges == env_copy_state.circuit.number_of_edges}")
    print(f"Pass copy test 1.8: {env_state.circuit.edge_from == env_copy_state.circuit.edge_from}")
    print(f"Pass copy test 1.9: {env_state.circuit.edge_to == env_copy_state.circuit.edge_to}")
    print(f"Pass copy test 1.10: {env_state.circuit.edge_reversed == env_copy_state.circuit.edge_reversed}")
    print(f"Pass copy test 1.11: {env_state.circuit.edge_logical_idx == env_copy_state.circuit.edge_logical_idx}")
    print(f"Pass copy test 1.12: {env_state.circuit.edge_physical_idx == env_copy_state.circuit.edge_physical_idx}")
    print(f"Pass copy test 1.13: {env_state.device_edges_list == env_copy_state.device_edges_list}")
    print(f"Pass copy test 1.14: {env_state.logical2physical_mapping == env_copy_state.logical2physical_mapping}")
    print(f"Pass copy test 1.15: {env_state.physical2logical_mapping == env_copy_state.physical2logical_mapping}")


def test_step(env_state2, env_copy_state2):
    print(f"Pass copy test 2.1: {env_state2.circuit.number_of_nodes == env_copy_state2.circuit.number_of_nodes}")
    print(f"Pass copy test 2.2: {env_state2.circuit.node_id == env_copy_state2.circuit.node_id}")
    print(f"Pass copy test 2.3: {env_state2.circuit.is_input == env_copy_state2.circuit.is_input}")
    print(f"Pass copy test 2.4: {env_state2.circuit.input_logical_idx == env_copy_state2.circuit.input_logical_idx}")
    print(f"Pass copy test 2.5: {not env_state2.circuit.input_physical_idx == env_copy_state2.circuit.input_physical_idx}")
    print(f"Pass copy test 2.6: {env_state2.circuit.node_type == env_copy_state2.circuit.node_type}")
    print(f"Pass copy test 2.7: {env_state2.circuit.number_of_edges == env_copy_state2.circuit.number_of_edges}")
    print(f"Pass copy test 2.8: {env_state2.circuit.edge_from == env_copy_state2.circuit.edge_from}")
    print(f"Pass copy test 2.9: {env_state2.circuit.edge_to == env_copy_state2.circuit.edge_to}")
    print(f"Pass copy test 2.10: {env_state2.circuit.edge_reversed == env_copy_state2.circuit.edge_reversed}")
    print(f"Pass copy test 2.11: {env_state2.circuit.edge_logical_idx == env_copy_state2.circuit.edge_logical_idx}")
    print(f"Pass copy test 2.12: {not env_state2.circuit.edge_physical_idx == env_copy_state2.circuit.edge_physical_idx}")
    print(f"Pass copy test 2.13: {env_state2.device_edges_list == env_copy_state2.device_edges_list}")
    print(f"Pass copy test 2.14: {not env_state2.logical2physical_mapping == env_copy_state2.logical2physical_mapping}")
    print(f"Pass copy test 2.15: {not env_state2.physical2logical_mapping == env_copy_state2.physical2logical_mapping}")


def main():
    # initialize two envs
    env = PySimpleSearchEnv(qasm_file_path="../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
                            backend_type_str="IBM_Q27_FALCON", seed=0, start_from_internal_prob=0,
                            initial_mapping_file_path="./mapping.txt")
    env_copy = env.copy()
    print("Passed copy test 0!")

    # check original state difference
    env_state = env.get_state()
    env_copy_state = env_copy.get_state()
    test_same(env_state, env_copy_state)
    print()

    for _ in range(100):
        # check state after step
        env.step(env.get_action_space()[0])
        env_state2 = env.get_state()
        env_copy_state2 = env_copy.get_state()
        test_step(env_state2, env_copy_state2)
        print()
        test_same(env_copy_state2, env_state)
        print()


if __name__ == '__main__':
    main()
