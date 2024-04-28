import argparse
from onpolicy.envs.quartz_initial_mapping.quartz_initial_env import SimpleInitialEnv


def main():
    # initialize
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.episode_length = 100
    env = SimpleInitialEnv(qasm_file_name="../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
                           backend_name="IBM_Q27_FALCON", all_args=args, max_obs_length=5000)
    print(f"{env.cur_qiskit_cost=}")
    env.reset()
    print(f"{env.cur_qiskit_cost=}")


if __name__ == '__main__':
    main()
