import os
import argparse
import string


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit_name", type=str, required=True)
    parser.add_argument("--device_name", type=str, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    args = parser.parse_args()

    # get mapping from target files
    initial_mapping_folder = f"./experiment/{args.circuit_name}/{args.device_name}/initial_mapping_dir"
    filename_list = os.listdir(initial_mapping_folder)
    mapping_str_list = []
    for filename in filename_list:
        # check gate count
        gate_count = int(filename.split("_")[0])
        if gate_count > args.threshold:
            continue

        # convert to C++ format
        with open(initial_mapping_folder + filename) as file_handle:
            mapping = file_handle.read()
            mapping_str = f"{mapping}".replace("[", "{").replace("]", "}")
            mapping_str_list.append(mapping_str)

    # log to terminal
    header_str = "std::vector<std::vector<int>> logical2physical = \n{"
    content = ", \n".join(mapping_str_list)
    tail_str = "};"
    print(header_str + content + tail_str)


if __name__ == '__main__':
    main()
