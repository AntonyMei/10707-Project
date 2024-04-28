from quartz import PySimpleHybridEnv


def main():
    env = PySimpleHybridEnv(
        # basic parameters
        qasm_file_path="../onpolicy/scripts/qasm_files/gf2^E5_mult_after_heavy.qasm",
        backend_type_str="IBM_Q27_FALCON",
        initial_mapping_file_path=f"./mapping.txt",
        # randomness and buffer
        seed=1,
        start_from_internal_prob=0.8,
        game_buffer_size=5,
        save_interval=3,
        # GameHybrid settings
        initial_phase_len=5,
        allow_nop_in_initial=True,
        initial_phase_reward=-1,
    )
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))
    print(env.step(env.get_action_space()[1]))


if __name__ == '__main__':
    main()
