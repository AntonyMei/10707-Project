import numpy as np
from quartz import PySimpleHybridEnv

from onpolicy.utils.quartz_utils import flatten_action, restore_action, py_action_list_2_list
from onpolicy.utils.quartz_utils import flatten_observation, restore_observation


def main():
    # initialize
    env = PySimpleHybridEnv(
        qasm_file_path="test.qasm", backend_type_str="IBM_Q27_FALCON",
        initial_mapping_file_path="./metirc_correlation/mapping.txt",
        seed=0, start_from_internal_prob=0,
        initial_phase_len=3, allow_nop_in_initial=True, initial_phase_reward=-0.3
    )
    print(f"make one step")
    env.step(env.get_action_space()[1])  # action 0 is nop that terminates phase 1
    print(f"make one step")
    env.step(env.get_action_space()[1])  # action 0 is nop that terminates phase 1

    # extract original state
    state = env.get_state()
    graph_state = state.circuit
    device_edge_list = state.device_edges_list
    logical2physical = state.logical2physical_mapping
    physical2logical = state.physical2logical_mapping
    is_initial_phase = state.is_initial_phase
    action_space = env.get_action_space()
    print(is_initial_phase)

    # record original state properties
    original_graph_state_number_of_nodes = graph_state.number_of_nodes
    original_graph_state_node_id = graph_state.node_id
    original_graph_state_is_input = graph_state.is_input
    original_graph_state_input_logical_idx = graph_state.input_logical_idx
    original_graph_state_input_physical_idx = graph_state.input_physical_idx
    original_graph_state_node_type = graph_state.node_type
    original_graph_state_number_of_edges = graph_state.number_of_edges
    original_graph_state_edge_from = graph_state.edge_from
    original_graph_state_edge_to = graph_state.edge_to
    original_graph_state_edge_reversed = graph_state.edge_reversed
    original_graph_state_edge_logical_idx = graph_state.edge_logical_idx
    original_graph_state_edge_physical_idx = graph_state.edge_physical_idx
    original_device_edge_list = device_edge_list
    original_logical2physical = logical2physical
    original_physical2logical = physical2logical
    original_action_space = py_action_list_2_list(action_space)
    original_is_initial_phase = is_initial_phase

    # serialize and deserialize
    flattened_obs = flatten_observation(graph_state=graph_state,
                                        device_edge_list=device_edge_list,
                                        logical2physical=logical2physical,
                                        physical2logical=physical2logical,
                                        action_space=action_space,
                                        is_initial_phase=is_initial_phase,
                                        target_length=1000,)
    print(f"flattened_obs size: {flattened_obs.size}")
    graph_state, device_edge_list, logical2physical, physical2logical, action_space, is_initial_phase = \
        restore_observation(flattened_obs=flattened_obs)

    # record deserialized state properties
    deserialized_graph_state_number_of_nodes = graph_state.number_of_nodes
    deserialized_graph_state_node_id = graph_state.node_id
    deserialized_graph_state_is_input = graph_state.is_input
    deserialized_graph_state_input_logical_idx = graph_state.input_logical_idx
    deserialized_graph_state_input_physical_idx = graph_state.input_physical_idx
    deserialized_graph_state_node_type = graph_state.node_type
    deserialized_graph_state_number_of_edges = graph_state.number_of_edges
    deserialized_graph_state_edge_from = graph_state.edge_from
    deserialized_graph_state_edge_to = graph_state.edge_to
    deserialized_graph_state_edge_reversed = graph_state.edge_reversed
    deserialized_graph_state_edge_logical_idx = graph_state.edge_logical_idx
    deserialized_graph_state_edge_physical_idx = graph_state.edge_physical_idx
    deserialized_device_edge_list = device_edge_list
    deserialized_logical2physical = logical2physical
    deserialized_physical2logical = physical2logical
    deserialized_action_space = action_space
    deserialized_is_initial_phase = is_initial_phase

    # check if deserialized state is the same as original state and print result
    print("original_graph_state_number_of_nodes == deserialized_graph_state_number_of_nodes: ",
          original_graph_state_number_of_nodes == deserialized_graph_state_number_of_nodes)
    print("original_graph_state_node_id == deserialized_graph_state_node_id: ",
          all(original_graph_state_node_id == deserialized_graph_state_node_id))
    print("original_graph_state_is_input == deserialized_graph_state_is_input: ",
          all(original_graph_state_is_input == deserialized_graph_state_is_input))
    print("original_graph_state_input_logical_idx == deserialized_graph_state_input_logical_idx: ",
          all(original_graph_state_input_logical_idx == deserialized_graph_state_input_logical_idx))
    print("original_graph_state_input_physical_idx == deserialized_graph_state_input_physical_idx: ",
          all(original_graph_state_input_physical_idx == deserialized_graph_state_input_physical_idx))
    print("original_graph_state_node_type == deserialized_graph_state_node_type: ",
          all(original_graph_state_node_type == deserialized_graph_state_node_type))
    print("original_graph_state_number_of_edges == deserialized_graph_state_number_of_edges: ",
          original_graph_state_number_of_edges == deserialized_graph_state_number_of_edges)
    print("original_graph_state_edge_from == deserialized_graph_state_edge_from: ",
          all(original_graph_state_edge_from == deserialized_graph_state_edge_from))
    print("original_graph_state_edge_to == deserialized_graph_state_edge_to: ",
          all(original_graph_state_edge_to == deserialized_graph_state_edge_to))
    print("original_graph_state_edge_reversed == deserialized_graph_state_edge_reversed: ",
          all(original_graph_state_edge_reversed == deserialized_graph_state_edge_reversed))
    print("original_graph_state_edge_logical_idx == deserialized_graph_state_edge_logical_idx: ",
          all(original_graph_state_edge_logical_idx == deserialized_graph_state_edge_logical_idx))
    print("original_graph_state_edge_physical_idx == deserialized_graph_state_edge_physical_idx: ",
          all(original_graph_state_edge_physical_idx == deserialized_graph_state_edge_physical_idx))
    print("original_device_edge_list == deserialized_device_edge_list: ",
          (np.array(original_device_edge_list) == deserialized_device_edge_list).all())
    print("original_logical2physical == deserialized_logical2physical: ",
          all(list(original_logical2physical.values()) == deserialized_logical2physical))
    print("original_physical2logical == deserialized_physical2logical: ",
          all(list(original_physical2logical.values()) == deserialized_physical2logical))
    print("original_action_space == deserialized_action_space: ",
          original_action_space == deserialized_action_space)
    print("original_is_initial_phase == deserialized_is_initial_phase: ",
          original_is_initial_phase == deserialized_is_initial_phase)

    # test action serdes
    print(f"{restore_action(flatten_action(10, 7)) == (10, 7)=}")


if __name__ == '__main__':
    main()
