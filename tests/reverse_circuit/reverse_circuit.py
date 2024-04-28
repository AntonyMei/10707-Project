import os

from qiskit.circuit import QuantumCircuit


def reverse_circuit(circuit_path, save_path):
    circuit = QuantumCircuit.from_qasm_file(circuit_path)
    reversed_circuit = circuit.inverse()
    # circuit.draw(output='mpl', filename="1.jpg")
    # reversed_circuit.draw(output='mpl', filename="2.jpg")
    reversed_circuit.qasm(filename=save_path)


def main():
    target_dir = "../../onpolicy/scripts/qasm_files"
    for file_name in os.listdir(target_dir):
        save_file_name = file_name.split(".")[0] + "_reversed.qasm"
        reverse_circuit(target_dir + "/" + file_name, target_dir + "/" + save_file_name)
        print(f"Finished Circuit: {file_name}.")


if __name__ == '__main__':
    main()
