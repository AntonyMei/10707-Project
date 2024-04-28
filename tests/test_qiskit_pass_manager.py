from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import SabreLayout, SabreSwap, SetLayout
from qiskit.transpiler.passmanager_config import PassManagerConfig


def sabre_pass_manager(pass_manager_config: PassManagerConfig) -> PassManager:
    coupling_map = pass_manager_config.coupling_map
    initial_layout = pass_manager_config.initial_layout
    layout_method = pass_manager_config.layout_method or "dense"
    routing_method = pass_manager_config.routing_method or "stochastic"

    # 1. sabre layout for initial mapping
    assert layout_method == "sabre"
    _improve_layout = SabreLayout(coupling_map, max_iterations=4, seed=0)

    # 2. apply layout
    _embed = [FullAncillaAllocation(coupling_map), EnlargeWithAncilla(), ApplyLayout()]

    # 3. sabre swap for routing
    assert routing_method == "sabre"
    _swap = [BarrierBeforeFinalMeasurements()]
    _swap += [SabreSwap(coupling_map, heuristic="decay", seed=5)]

    # Build pass manager
    pm1 = PassManager()
    if initial_layout is None:
        print("!!!! Start from sabre layout !!!!")
        pm1.append(_improve_layout)
        pm1.append(_embed)
    else:
        print("!!!! Start from GIVEN layout !!!!")
        _given_layout = SetLayout(initial_layout)
        pm1.append(_given_layout)
        pm1.append([ApplyLayout()])

    pm1.append(_swap)

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


def run_benchmark(qasm_file_name, device_name):
    # get device coupling map
    if device_name == "IBM_Q20_Tokyo":
        coupling_map = IBM_Q20_Tokyo()
    elif device_name == "IBM_Q27_Falcon":
        coupling_map = IBM_Q27_Falcon()
    else:
        raise NotImplementedError

    # parse qasm file into a circuit
    circuit = QuantumCircuit(20)
    with open("../onpolicy/scripts/qasm_files/" + qasm_file_name) as file:
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

    # run sabre layout and sabre swap
    sabre_manager = sabre_pass_manager(PassManagerConfig(coupling_map=coupling_map, layout_method="sabre",
                                                         routing_method="sabre"))
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

    # ***************************************************************************** #
    # this is the layout found by sabre layout
    layout = sabre_manager.passes()[0]["passes"][0].property_set["layout"]
    logical2physical = layout.get_virtual_bits()
    print(logical2physical)
    # set layout and run again
    new_manager = sabre_pass_manager(PassManagerConfig(coupling_map=coupling_map,
                                                       layout_method="sabre",
                                                       routing_method="sabre",
                                                       initial_layout=layout))
    new_sabre_circuit = new_manager.run(circuit)
    new_layout = new_manager.passes()[0]["passes"][0].property_set["layout"]
    new_op_list = dict(new_sabre_circuit.count_ops())
    swap_count = new_op_list["swap"]
    print(new_layout.get_virtual_bits())

    print(f"1st run swap count: {sabre_swap_count}, restart swap count: {swap_count}.")
    # ***************************************************************************** #

    return ori_gate_count, sabre_gate_count, sabre_swap_count


def main():
    # parameters
    # gf2^E5_mult_after_heavy.qasm, barenco_tof_10_before.qasm, csla_mux_3_after_heavy.qasm
    # IBM_Q20_Tokyo, IBM_Q27_Falcon
    benchmark_runs = 1
    qasm_file_name = "gf2^E5_mult_after_heavy.qasm"
    device_name = "IBM_Q20_Tokyo"

    # run benchmark
    original_gate_count_list = []
    sabre_gate_count_list, sabre_swap_count_list = [], []
    for _ in range(benchmark_runs):
        data = run_benchmark(qasm_file_name=qasm_file_name, device_name=device_name)
        ori_gate_count, sabre_gate_count, sabre_swap_count = data
        original_gate_count_list.append(ori_gate_count)
        sabre_gate_count_list.append(sabre_gate_count)
        sabre_swap_count_list.append(sabre_swap_count)


if __name__ == '__main__':
    main()
