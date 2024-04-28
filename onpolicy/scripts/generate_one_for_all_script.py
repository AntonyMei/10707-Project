import os


def main():
    # get scripts to run
    command_list = []
    for file_name in os.listdir("./"):
        if "slurm" in file_name and "sh" in file_name:
            command_list.append(f"sbatch {file_name}")
    print(len(command_list))

    # divide command list into 5 groups and write to 5 scripts
    num_scripts = 1
    for i in range(num_scripts):
        with open(f"final_run_{i}.sh", "w") as script_file:
            start = (len(command_list) + num_scripts - 1) // num_scripts * i
            end = min((len(command_list) + num_scripts - 1) // num_scripts * (i + 1), len(command_list))
            for command in command_list[start:end]:
                script_file.write(f"{command}\n")
            script_file.write("wait\n")
        print(f"Info: Generated final_run_{i}.sh!")


if __name__ == '__main__':
    main()
