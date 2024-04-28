from typing import List, Tuple
from quartz import PyGraphState, PyAction, PyState

import os
import numpy as np
import dgl
import torch

from qiskit import QuantumCircuit
from qiskit.transpiler import Layout, PassManager, CouplingMap
from qiskit.transpiler.passes import ApplyLayout, SabreSwap, SetLayout


def profile(func):
    from line_profiler import LineProfiler

    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return result

    return wrapper


def py_action_list_2_list(action_list: List[PyAction]):
    decoded_actions = []
    for action in action_list:
        decoded_actions.append((action.qubit_idx_0, action.qubit_idx_1))
    return decoded_actions


def flatten_observation(graph_state, device_edge_list, logical2physical,
                        physical2logical, action_space, is_initial_phase, target_length):
    """
    flattened format: graph_state.number_of_nodes, graph_state.number_of_edges, flatten array
                    + device edge count, edge0_idx0, edge1_idx0, ..., edge0_idx1, edge1_idx1, ...
                    + logical2physical (flatten as array)
                    + physical2logical (flatten as array)
                    + action space: action0_idx0, action1_idx0, ..., action0_idx1, action1_idx1, ...
    """
    # flatten graph state
    flattened_graph_state = np.array([graph_state.number_of_nodes, graph_state.number_of_edges])
    flattened_graph_state = np.concatenate([flattened_graph_state, graph_state.node_id,
                                            graph_state.is_input, graph_state.input_logical_idx,
                                            graph_state.input_physical_idx, graph_state.node_type,
                                            graph_state.edge_from, graph_state.edge_to,
                                            graph_state.edge_reversed, graph_state.edge_logical_idx,
                                            graph_state.edge_physical_idx])

    # flatten device edge list
    device_edge_list = np.array(device_edge_list)
    flattened_device = np.array([len(device_edge_list)])
    flattened_device = np.concatenate([flattened_device, device_edge_list[:, 0], device_edge_list[:, 1]])

    # flatten mapping table
    assert len(logical2physical) == len(physical2logical)
    logical2physical_list = []
    physical2logical_list = []
    for i in range(len(logical2physical)):
        logical2physical_list.append(logical2physical[i])
        physical2logical_list.append(physical2logical[i])
    flattened_mapping = np.array([len(logical2physical)])
    flattened_mapping = np.concatenate([flattened_mapping, logical2physical_list, physical2logical_list])

    # flatten action space
    flattened_action_space = np.array([len(action_space)])
    decoded_action_space = np.array(py_action_list_2_list(action_space))
    flattened_action_space = np.concatenate([flattened_action_space, decoded_action_space[:, 0],
                                             decoded_action_space[:, 1]])

    # flatten is_initial_state
    flattened_is_initial_phase = np.array([is_initial_phase])

    # concat together and return
    result = np.concatenate([flattened_graph_state, flattened_device, flattened_mapping,
                             flattened_action_space, flattened_is_initial_phase])
    result = np.concatenate([[result.size + 1], result])
    if target_length is not None:
        assert target_length > result.size, f"Target length {target_length} should be larger than" \
                                            f" flattened obs size {result.size}"
        padding = np.zeros(target_length - result.size, dtype=int)
        result = np.concatenate([result, padding])
    return result


def restore_observation(flattened_obs):
    flattened_obs = np.asarray(flattened_obs, dtype=int)
    total_length = flattened_obs[0]
    # restore graph state
    graph_state = PyGraphState()

    graph_state.number_of_nodes = flattened_obs[1]
    graph_state.number_of_edges = flattened_obs[2]
    cur_pos_pointer = 3

    graph_state_node_data = flattened_obs[cur_pos_pointer: cur_pos_pointer + graph_state.number_of_nodes * 5]
    graph_state_node_data = graph_state_node_data.reshape([5, graph_state.number_of_nodes])
    graph_state.node_id = graph_state_node_data[0]
    graph_state.is_input = np.array(graph_state_node_data[1], dtype=bool)
    graph_state.input_logical_idx = graph_state_node_data[2]
    graph_state.input_physical_idx = graph_state_node_data[3]
    graph_state.node_type = graph_state_node_data[4]
    cur_pos_pointer += graph_state.number_of_nodes * 5

    graph_state_edge_data = flattened_obs[cur_pos_pointer: cur_pos_pointer + graph_state.number_of_edges * 5]
    graph_state_edge_data = graph_state_edge_data.reshape([5, graph_state.number_of_edges])
    graph_state.edge_from = graph_state_edge_data[0]
    graph_state.edge_to = graph_state_edge_data[1]
    graph_state.edge_reversed = np.array(graph_state_edge_data[2], dtype=bool)
    graph_state.edge_logical_idx = graph_state_edge_data[3]
    graph_state.edge_physical_idx = graph_state_edge_data[4]
    cur_pos_pointer += graph_state.number_of_edges * 5

    # restore device edge list
    device_edge_count = flattened_obs[cur_pos_pointer]
    cur_pos_pointer += 1
    device_edge_data = flattened_obs[cur_pos_pointer: cur_pos_pointer + device_edge_count * 2]
    device_edge_data = device_edge_data.reshape([2, device_edge_count])
    # device_edge_list = []
    # for i in range(device_edge_count):
    #     device_edge_list.append([device_edge_data[0][i], device_edge_data[1][i]])
    device_edge_list = device_edge_data.T   # faster (shape=(device_edge_count, 2))
    cur_pos_pointer += device_edge_count * 2

    # restore mapping table
    mapping_table_size = flattened_obs[cur_pos_pointer]
    cur_pos_pointer += 1
    logical2physical = flattened_obs[cur_pos_pointer: cur_pos_pointer + mapping_table_size]
    cur_pos_pointer += mapping_table_size
    physical2logical = flattened_obs[cur_pos_pointer: cur_pos_pointer + mapping_table_size]
    cur_pos_pointer += mapping_table_size
    # we do not convert to map here for better performance
    # logical2physical = {}
    # physical2logical = {}
    # for i in range(mapping_table_size):
    #     logical2physical[i] = logical2physical_data[i]
    #     physical2logical[i] = physical2logical_data[i]

    # restore action space
    action_space_size = flattened_obs[cur_pos_pointer]
    cur_pos_pointer += 1
    action_space_data = flattened_obs[cur_pos_pointer: cur_pos_pointer + 2 * action_space_size]
    action_space_data = action_space_data.reshape([2, action_space_size]).transpose()
    action_space = []
    for action in action_space_data:
        action_space.append(tuple(action))
    cur_pos_pointer += 2 * action_space_size

    # restore is_initial_state
    is_initial_phase = bool(flattened_obs[cur_pos_pointer])
    cur_pos_pointer += 1

    # return
    assert cur_pos_pointer == total_length
    return graph_state, device_edge_list, logical2physical, physical2logical, action_space, is_initial_phase


def flatten_action(qubit_idx_0, qubit_idx_1):
    return qubit_idx_0 * 2000 + qubit_idx_1


def restore_action(flattened_action):
    flattened_action = int(flattened_action)
    qubit_idx_0 = int(flattened_action / 2000)
    qubit_idx_1 = flattened_action % 2000
    return qubit_idx_0, qubit_idx_1


def graph_state_2_dgl(graph_state: PyGraphState, device):
    g = dgl.graph((torch.tensor(graph_state.edge_from, dtype=torch.int32, device=device),
                   torch.tensor(graph_state.edge_to, dtype=torch.int32, device=device)))
    g.edata["logical_idx"] = torch.tensor(graph_state.edge_logical_idx, dtype=torch.int32, device=device)
    g.edata["physical_idx"] = torch.tensor(graph_state.edge_physical_idx, dtype=torch.int32, device=device)
    g.edata["reversed"] = torch.tensor(graph_state.edge_reversed, dtype=torch.int32, device=device)
    g.ndata["is_input"] = torch.tensor(graph_state.is_input, dtype=torch.int32, device=device)
    return g


def graph_state_2_dgl_batch(
        graph_state_batch: PyGraphState,
        device: torch.device = torch.device("cpu")
) -> Tuple[dgl.DGLHeteroGraph, List[int], List[torch.IntTensor]]:
    g_list: List[dgl.DGLHeteroGraph] = []
    qubit_count_list: List[int] = []
    edge_physical_idx_list: List[torch.LongTensor] = []
    for graph_state in graph_state_batch:
        qubit_count_list.append(int(torch.sum(torch.tensor(graph_state.is_input))))
        g = dgl.graph((
            graph_state.edge_from, graph_state.edge_to,
        ))
        g.edata["logical_idx"] = torch.tensor(graph_state.edge_logical_idx, dtype=torch.int32)
        edge_physical_idx = torch.IntTensor(graph_state.edge_physical_idx)
        edge_physical_idx_list.append(edge_physical_idx)
        g.edata["physical_idx"] = edge_physical_idx
        g.edata["reversed"] = torch.IntTensor(graph_state.edge_reversed)
        g.ndata["is_input"] = torch.IntTensor(graph_state.is_input)
        g_list.append(g)
    # for end
    circuit_dgl_batch = dgl.batch(g_list).to(device)
    return circuit_dgl_batch, qubit_count_list, edge_physical_idx_list


def device_edges_2_dgl(device_edges_list, device):
    # pack device edges into tensor and create dgl graph
    src_id = []
    dst_id = []
    for edge in device_edges_list:
        assert len(edge) == 2
        src_id.append(edge[0])
        dst_id.append(edge[1])
    dgl_graph = dgl.graph((torch.tensor(src_id, dtype=torch.int32, device=device),
                           torch.tensor(dst_id, dtype=torch.int32, device=device)))

    # add features to the new graph
    node_degree = [0] * dgl_graph.number_of_nodes()
    node_id = list(range(dgl_graph.number_of_nodes()))
    for edge in device_edges_list:
        node_degree[edge[0]] += 1
    dgl_graph.ndata["degree"] = torch.tensor(node_degree, dtype=torch.int32, device=device)
    dgl_graph.ndata["id"] = torch.tensor(node_id, dtype=torch.int32, device=device)
    return dgl_graph


def device_edges_2_dgl_batch(
        device_edge_list_batch: List[List[int]],
        device: torch.device = torch.device("cpu"),
        dummy: bool = False
) -> dgl.DGLHeteroGraph:
    device_edge_list_batch = torch.tensor(np.array(device_edge_list_batch), dtype=torch.int32)
    dgl_graph_list: List[dgl.DGLHeteroGraph] = []
    if dummy:
        # in dummy mode, the device graphs are the same, so we only need to reconstruct once
        batch_size = len(device_edge_list_batch)
        dgl_graph = dgl.graph((
            device_edge_list_batch[0][:, 0], device_edge_list_batch[0][:, 1]  # src, dst
        ))
        num_nodes = dgl_graph.num_nodes()
        dgl_graph.ndata["id"] = torch.arange(num_nodes, dtype=torch.int32)
        node_degrees = torch.bincount(device_edge_list_batch[0][:, 0])
        dgl_graph.ndata["degree"] = node_degrees
        dgl_graph_list.append(dgl_graph)
        dgl_graph_list *= batch_size
    else:
        for device_edge_list in device_edge_list_batch:
            dgl_graph = dgl.graph((
                device_edge_list[:, 0], device_edge_list[:, 1]  # src, dst
            ))
            num_nodes = dgl_graph.num_nodes()
            dgl_graph.ndata["id"] = torch.arange(num_nodes, dtype=torch.int32)
            node_degrees = torch.bincount(device_edge_list[:, 0])
            dgl_graph.ndata["degree"] = node_degrees
            dgl_graph_list.append(dgl_graph)
    # end for
    dgl_graph_batch = dgl.batch(dgl_graph_list).to(device)
    return dgl_graph_batch


def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def log_model_weights(model, rank):
    print("*************************************")
    for name, param in model.named_parameters():
        print(name)
        if len(param.shape) == 1:
            print(f"[rank {rank}] Weight", param[0:5])
            if param.requires_grad:
                print(f"[rank {rank}] Gradient:", param.grad[0:5])
        elif len(param.shape) == 2:
            print(f"[rank {rank}] Weight", param[0][0:5])
            if param.requires_grad:
                print(f"[rank {rank}] Gradient:", param.grad[0][0:5])
        else:
            assert False
        print()
    print("*************************************")


def log_model_weights_norm(model, rank):
    print("*************************************")
    for name, param in model.named_parameters():
        print(name)
        print(f"[rank {rank}] Weight norm:", param.norm())
        if param.requires_grad:
            print(f"[rank {rank}] Gradient norm:", param.grad.norm())
        print()
    print("*************************************")


def qiskit_parse_gate_type(tp, circuit):
    if tp == "cx":
        return circuit.cx
    elif tp == "h":
        return circuit.h
    elif tp == "t":
        return circuit.t
    elif tp == "tdg":
        return circuit.tdg
    elif tp == "s":
        return circuit.s
    elif tp == "sdg":
        return circuit.sdg
    else:
        raise NotImplementedError


def build_qiskit_circuit(qasm_file_path, num_regs):
    # parse qasm file into a circuit
    circuit = QuantumCircuit(num_regs)
    with open(qasm_file_path) as file:
        # omit the header
        file.readline()
        file.readline()
        line = file.readline()
        num_qubits = int(line.split(' ')[1].split(']')[0].split('[')[1])
        # parse the rest
        line = file.readline()
        while line != '':
            # add to circuit
            arg_list = line.split(' ')
            if arg_list[0] == '':
                arg_list = arg_list[1:]
            if len(arg_list) == 3:
                # gate type
                tp = arg_list[0]
                # two qubits gate
                qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
                qubit2 = int(arg_list[2].split(']')[0].split('[')[1])
                qiskit_parse_gate_type(tp=tp, circuit=circuit)(qubit1, qubit2)
            elif len(arg_list) == 2:
                # gate type
                tp = arg_list[0]
                # single qubit gate
                qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
                qiskit_parse_gate_type(tp=tp, circuit=circuit)(qubit1)
            else:
                assert False
            # read another line
            line = file.readline()
    return circuit


def get_qiskit_sabre_cost(state: PyState, circuit: QuantumCircuit, coupling_map: CouplingMap):
    # build logical -> physical mapping from state
    logical2physical_mapping = state.logical2physical_mapping
    logical2physical_list = []
    for i in range(len(logical2physical_mapping)):
        logical2physical_list.append(logical2physical_mapping[i])
    initial_layout = Layout.from_intlist(logical2physical_list, *circuit.qregs)

    # build pass manager
    pass_manager = PassManager()
    pass_manager.append(SetLayout(initial_layout))
    pass_manager.append(ApplyLayout())
    pass_manager.append(SabreSwap(coupling_map, heuristic="decay", seed=0))

    # get final result
    result_circuit = pass_manager.run(circuit)
    swap_count = dict(result_circuit.count_ops())["swap"]
    return swap_count


def get_latest_checkpoint_id(path):
    filename_list = os.listdir(path)
    latest_epoch = None
    for filename in filename_list:
        if "model" not in filename:
            continue
        cur_model_epoch = int(filename.split(".")[0].split("_")[1])
        if latest_epoch is None or cur_model_epoch > latest_epoch:
            latest_epoch = cur_model_epoch

    assert latest_epoch is not None, "Empty directory for checkpoint!"
    return latest_epoch
