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

        # generate slurm conf file & slurm run script only if this is a forward file
        # conf file
        if not is_backward:
            with open(f"./conf_{circuit_name}.conf", "w") as script_file:
                script_file.write(f"0 bash asplos2024_q65_{circuit_name}_forward.sh\n")
                script_file.write(f"1 bash asplos2024_q65_{circuit_name}_backward.sh\n")
            with open(f"./slurm_{circuit_name}.sh", "w") as script_file:
                script_file.write(f"#!/bin/bash\n"
                                  f"#SBATCH -A m4138\n"
                                  f"#SBATCH -t 24:00:00\n"
                                  f"#SBATCH --gpus-per-node=4\n"
                                  f"#SBATCH -C \"gpu\"\n"
                                  f"#SBATCH -c 128\n"
                                  f"#SBATCH -q regular\n"
                                  f"#SBATCH -n 2\n"
                                  f"#SBATCH --output=output_{circuit_name}.txt\n"
                                  f"srun -n1 bash asplos2024_q65_{circuit_name}_forward.sh &\n"
                                  f"srun -n1 bash asplos2024_q65_{circuit_name}_backward.sh &\n"
                                  f"wait"
                                  )
                # script_file.write(f"srun -A m4138 -t 12:00:00 --gpus-per-node=4 -C \"gpu\" -c 128 -q regular -n 2"
                #                   f" --multi-prog conf_{circuit_name}.conf")


if __name__ == '__main__':
    main()
