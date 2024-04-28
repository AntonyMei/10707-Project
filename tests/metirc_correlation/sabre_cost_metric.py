import random

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import SabreSwap, SetLayout
from qiskit.transpiler.passmanager_config import PassManagerConfig


def sabre_pass_manager(pass_manager_config: PassManagerConfig, seed) -> PassManager:
    # extract parameters
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    routing_method = pass_manager_config.routing_method

    # Build pass manager
    pm1 = PassManager()

    # 1. set layout
    assert initial_layout is not None
    _given_layout = SetLayout(initial_layout)
    _apply_layout = ApplyLayout()
    pm1.append(_given_layout)
    pm1.append(_apply_layout)

    # 2. sabre swap for routing
    assert routing_method == "sabre"
    _swap = SabreSwap(coupling_map, heuristic="decay", seed=seed)
    pm1.append(_swap)

    # return
    return pm1


def parse_gate_type(tp, circuit):
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


def IBM_Q20_Tokyo():
    # identical to IBM Q20 Tokyo
    coupling = [
        # rows
        [0, 1], [1, 2], [2, 3], [3, 4],
        [5, 6], [6, 7], [7, 8], [8, 9],
        [10, 11], [11, 12], [12, 13], [13, 14],
        [15, 16], [16, 17], [17, 18], [18, 19],
        # cols
        [0, 5], [5, 10], [10, 15],
        [1, 6], [6, 11], [11, 16],
        [2, 7], [7, 12], [12, 17],
        [3, 8], [8, 13], [13, 18],
        [4, 9], [9, 14], [14, 19],
        # crossings
        [1, 7], [2, 6],
        [3, 9], [4, 8],
        [5, 11], [6, 10],
        [8, 12], [7, 13],
        [11, 17], [12, 16],
        [13, 19], [14, 18]
    ]
    reversed_coupling = []
    for pair in coupling:
        reversed_coupling.append([pair[1], pair[0]])
    coupling_map = CouplingMap(couplinglist=coupling + reversed_coupling)
    return coupling_map


def IBM_Q27_Falcon():
    # identical to IBM Q27 Falcon
    coupling = [
        # 1st row
        [0, 1], [1, 4], [4, 7], [7, 10], [10, 12], [12, 15], [15, 18],
        [18, 21], [21, 23],
        # 2nd row
        [3, 5], [5, 8], [8, 11], [11, 14], [14, 16], [16, 19], [19, 22],
        [22, 25], [25, 26],
        # cols
        [6, 7], [17, 18], [1, 2], [2, 3], [12, 13], [13, 14], [23, 24],
        [24, 25], [8, 9], [19, 20]
    ]
    reversed_coupling = []
    for pair in coupling:
        reversed_coupling.append([pair[1], pair[0]])
    coupling_map = CouplingMap(couplinglist=coupling + reversed_coupling)
    return coupling_map


def sabre_round(qasm_file_name, device_name, mapping_file_name, random_seed):
    """
    logical2physical: a dict {logical_idx (int) -> physical_idx (int)}
    """
    # get device coupling map
    if device_name == "IBM_Q20_Tokyo" or device_name == "IBM_Q20_TOKYO":
        coupling_map, reg_count = IBM_Q20_Tokyo(), 20
    elif device_name == "IBM_Q27_Falcon" or device_name == "IBM_Q27_FALCON":
        coupling_map, reg_count = IBM_Q27_Falcon(), 27
    else:
        raise NotImplementedError

    # parse qasm file into a circuit
    circuit = QuantumCircuit(reg_count)
    with open("../search/qasm_files/" + qasm_file_name) as file:
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
                parse_gate_type(tp=tp, circuit=circuit)(qubit1, qubit2)
            elif len(arg_list) == 2:
                # gate type
                tp = arg_list[0]
                # single qubit gate
                qubit1 = int(arg_list[1].split(']')[0].split('[')[1])
                parse_gate_type(tp=tp, circuit=circuit)(qubit1)
            else:
                assert False
            # read another line
            line = file.readline()

    # load initial mapping from file
    with open("./mappings/" + mapping_file_name, "r") as mapping_file:
        # read line and convert into mapping
        mapping_str = mapping_file.readline()
        mapping_list = mapping_str.split()
        logical2physical_list = [int(idx) for idx in mapping_list]

        # check there is no second line
        assert mapping_file.readline() == ""

    # change initial mapping into sabre format
    initial_layout = Layout.from_intlist(logical2physical_list, *circuit.qregs)

    # run sabre layout and sabre swap
    sabre_manager = sabre_pass_manager(PassManagerConfig(coupling_map=coupling_map,
                                                         initial_layout=initial_layout,
                                                         routing_method="sabre"),
                                       seed=random_seed)
    sabre_circuit = sabre_manager.run(circuit)

    # original gate count
    ori_circuit_op_list = dict(circuit.count_ops())
    ori_gate_count = 0
    for key in ori_circuit_op_list:
        if key == "swap":
            assert False, "swap in original circuit!"
        else:
            ori_gate_count += ori_circuit_op_list[key]

    # get gate count of sabre
    sabre_circuit_op_list = dict(sabre_circuit.count_ops())
    sabre_gate_count = 0
    sabre_swap_count = 0
    for key in sabre_circuit_op_list:
        if key == "swap":
            sabre_gate_count += 3 * sabre_circuit_op_list[key]
            sabre_swap_count = sabre_circuit_op_list[key]
        else:
            assert sabre_circuit_op_list[key] == ori_circuit_op_list[key]
            sabre_gate_count += sabre_circuit_op_list[key]
    assert sabre_gate_count - 3 * sabre_swap_count == ori_gate_count

    return ori_gate_count, sabre_gate_count, sabre_swap_count


def sabre_cost_metric(qasm_file_name, device_name, mapping_file_name,
                      round_number, random_seed):
    # random seed
    random.seed(random_seed)

    # run sabre experiments
    original_gate_count_list = []
    sabre_gate_count_list, sabre_swap_count_list = [], []
    for _ in range(round_number):
        data = sabre_round(qasm_file_name=qasm_file_name, device_name=device_name,
                           mapping_file_name=mapping_file_name, random_seed=random.randint(0, 9999))
        ori_gate_count, sabre_gate_count, sabre_swap_count = data
        original_gate_count_list.append(ori_gate_count)
        sabre_gate_count_list.append(sabre_gate_count)
        sabre_swap_count_list.append(sabre_swap_count)

    # post-processing on data
    assert len(set(original_gate_count_list)) == 1
    min_sabre_cost = min(sabre_gate_count_list)
    avg_sabre_cost = sum(sabre_gate_count_list) / round_number
    min_swap_count = min(sabre_swap_count_list)
    avg_swap_count = sum(sabre_swap_count_list) / round_number
    return min_sabre_cost, avg_sabre_cost, min_swap_count, avg_swap_count


def batch_sabre_cost_metric(qasm_file_name, device_name, mapping_file_name_batch,
                            round_number, random_seed):
    min_sabre_cost_list = []
    avg_sabre_cost_list = []
    min_swap_count_list = []
    avg_swap_count_list = []
    for mapping_file_name in mapping_file_name_batch:
        data = sabre_cost_metric(qasm_file_name=qasm_file_name, device_name=device_name,
                                 mapping_file_name=mapping_file_name, round_number=round_number,
                                 random_seed=random_seed)
        min_sabre_cost, avg_sabre_cost, min_swap_count, avg_swap_count = data
        min_sabre_cost_list.append(min_sabre_cost)
        avg_sabre_cost_list.append(avg_sabre_cost)
        min_swap_count_list.append(min_swap_count)
        avg_swap_count_list.append(avg_swap_count)
    return min_sabre_cost_list, avg_sabre_cost_list, min_swap_count_list, avg_swap_count_list


def test_sabre_metric():
    """
    This is only used in test.
    """
    qasm_file_name = "gf2^E5_mult_after_heavy.qasm"
    device_name = "IBM_Q27_Falcon"
    mapping_file_name = "initial_mapping.txt"
    random_seed = 0
    round_number = 16

    # run benchmark
    min_cost, avg_cost, min_swap, avg_swap = sabre_cost_metric(qasm_file_name=qasm_file_name, device_name=device_name,
                                                               mapping_file_name=mapping_file_name,
                                                               random_seed=random_seed,
                                                               round_number=round_number)
    print(f"Sabre min cost = {min_cost} ({min_swap} swaps), avg cost = {avg_cost} ({avg_swap} swaps).")


if __name__ == '__main__':
    test_sabre_metric()
