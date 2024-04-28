import os

device_abbr_map = {
    "IBM_Q27_FALCON": "q27",
    "IBM_Q65_HUMMINGBIRD": "q65",
    "IBM_Q127_EAGLE": "q127"
}


def main():
    # ******************* Parameters ******************* #
    seed = 0
    # ************************************************** #

    # iterate through the folder to generate scripts
    for filename in os.listdir("./qasm_files/"):
        # check whether we should generate script for backward agents or forward agents
        # (backward agent runs on reversed circuits)
        is_backward = "reversed" in filename

        # circuit and the reverse of this circuit
        circuit_name = filename.split(".")[0]
        if is_backward:
            reversed_circuit_name = circuit_name[:-9]
        else:
            reversed_circuit_name = circuit_name + "_reversed"
        qasm_file_name = circuit_name + ".qasm"
        reversed_qasm_file_name = reversed_circuit_name + ".qasm"

        # number of qubits determines which device it should run on
        num_qubits = None
        with open(f"./qasm_files/{filename}") as file:
            for line in file:
                if line.startswith("qreg"):
                    num_qubits = int(line.split('[')[1].split(']')[0])
                    break
                else:
                    continue
        assert num_qubits is not None
        device_name_list = ["IBM_Q65_HUMMINGBIRD"]

        # generate the scripts
        for device_name in device_name_list:
            device_abbr = device_abbr_map[device_name]
            experiment_name = reversed_circuit_name if is_backward else circuit_name
            script_prefix = "backward" if is_backward else "forward"
            with open(f"./asplos2024_{device_abbr}_{experiment_name}_{script_prefix}.sh", "w") as script_file:
                script_file.write(f"bash {script_prefix}_quartz_physical_train.sh {seed} {device_name} {qasm_file_name} {reversed_qasm_file_name}")
            print(f"Info: Generated {script_prefix} script for {experiment_name} on {device_name}!")
        print()


if __name__ == '__main__':
    main()
