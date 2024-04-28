#!/usr/bin/env python
import datetime
import shutil
import time

from onpolicy.envs.quartz_physical.quartz_env import SimpleHybridEnv
from onpolicy.envs.quartz_initial_mapping.quartz_initial_env import SimpleInitialEnv
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.runner.qiskit_sabre.qiskit_sabre import run_sabre

"""Train script for quartz."""


def make_train_env(all_args, ddp_rank, save_dir, save_threshold):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "quartz_physical":
                env = SimpleHybridEnv(qasm_file_name="qasm_files/" + all_args.qasm_file_name,
                                      backend_name=all_args.backend_name,
                                      all_args=all_args,
                                      env_seed=all_args.seed + rank * 10 + ddp_rank * 2000,
                                      ddp_rank=ddp_rank,
                                      max_obs_length=all_args.max_obs_length,
                                      is_eval=False,
                                      save_dir=save_dir,
                                      save_threshold=save_threshold)
            elif all_args.env_name == "quartz_initial":
                assert False, "Quartz Initial is obsolete"
                # env = SimpleInitialEnv(qasm_file_name="qasm_files/" + all_args.qasm_file_name,
                #                        backend_name=all_args.backend_name,
                #                        all_args=all_args,
                #                        max_obs_length=all_args.max_obs_length)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 10 + ddp_rank * 2000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, ddp_rank, save_dir, save_threshold):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "quartz_physical":
                env = SimpleHybridEnv(qasm_file_name="qasm_files/" + all_args.qasm_file_name,
                                      backend_name=all_args.backend_name,
                                      all_args=all_args,
                                      env_seed=all_args.seed + rank * 10 + ddp_rank * 2000 + 100000,
                                      ddp_rank=ddp_rank,
                                      max_obs_length=all_args.max_obs_length,
                                      is_eval=True,
                                      save_dir=save_dir,
                                      save_threshold=save_threshold)
            elif all_args.env_name == "quartz_initial":
                assert False, "Quartz Initial is obsolete"
                # env = SimpleInitialEnv(qasm_file_name="qasm_files/" + all_args.qasm_file_name,
                #                        backend_name=all_args.backend_name,
                #                        all_args=all_args,
                #                        max_obs_length=all_args.max_obs_length)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 10 + ddp_rank * 2000 + 100000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--world_size", type=int, required=True, help="DDP world size")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # parse arguments
    parser = get_config()
    all_args = parse_args(args, parser)

    # spawn training processes
    print("Info: rl agent version 1.0.3 - Full")
    mp.spawn(system_setup, args=(all_args,), nprocs=all_args.world_size)


def system_setup(rank, all_args):
    # check algorithm type
    assert all_args.algorithm_name == "quartz_ppo"

    # init ddp
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=all_args.world_size,
                            timeout=datetime.timedelta(seconds=10000))

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print(f"[rank {rank}] choose to use gpu (cuda:{rank})...")
        device = torch.device(f"cuda:{rank}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print(f"[rank {rank}] choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    if "reversed" in all_args.qasm_file_name:
        dir_suffix = all_args.reversed_qasm_file_name
    else:
        dir_suffix = all_args.qasm_file_name
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name / dir_suffix
    os.makedirs(str(run_dir), exist_ok=True)
    os.makedirs(f"./experiment/{all_args.qasm_file_name}/{all_args.backend_name}/eval_model_dir", exist_ok=True)
    os.makedirs(f"./experiment/{all_args.qasm_file_name}/{all_args.backend_name}/initial_mapping_dir", exist_ok=True)

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         mode="offline")
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # generate original set of initial mappings using qiskit sabre
    # only run SABRE on rank 0 && not running in two-way mode
    assert not all_args.two_way_mode == "none"
    if rank == 0:
        # rank 0 runs sabre and save the mappings for all ddp workers
        # TODO: maybe each worker can have its own mapping
        save_path_list = [f"./experiment/{all_args.qasm_file_name}/{all_args.backend_name}"
                          f"/initial_mapping_dir/mapping_rank{i}.txt" for i in range(all_args.world_size)]
        save_swap_count, _ = run_sabre(qasm_file_name=all_args.qasm_file_name, device_name=all_args.backend_name,
                                       num_runs=all_args.num_sabre_runs, num_saves=all_args.num_sabre_saves,
                                       save_path_list=save_path_list, seed=all_args.seed * all_args.num_sabre_runs)

        # log to wandb
        print(f"[Rank {rank}] Initial mappings saved for all ddp workers!")
        for idx, swap_count in enumerate(save_swap_count):
            wandb.log({"swap_count": swap_count}, step=idx)

    dist.barrier()
    time.sleep(0.5 * rank)

    # make dirs in wandb run dir for file saving
    os.makedirs(str(wandb.run.dir) + "/mappings")
    os.makedirs(str(wandb.run.dir) + "/latest_model")
    os.makedirs(str(wandb.run.dir) + "/plans")

    # backup initial sabre mapping in wandb
    shutil.copyfile(src=f"./experiment/{all_args.qasm_file_name}/{all_args.backend_name}/initial_mapping_dir"
                        f"/mapping_rank{rank}.txt",
                    dst=str(wandb.run.dir) + "/mappings/sabre_mappings.txt")

    # seed
    torch.manual_seed(all_args.seed + rank)
    torch.cuda.manual_seed_all(all_args.seed + rank)
    np.random.seed(all_args.seed + rank)

    # env init
    save_dir = str(wandb.run.dir) + "/plans"
    save_threshold = all_args.qasm_save_threshold
    envs = make_train_env(all_args, ddp_rank=rank, save_dir=save_dir, save_threshold=save_threshold)
    eval_envs = make_eval_env(all_args, ddp_rank=rank, save_dir=save_dir, save_threshold=save_threshold) \
        if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "rank": rank
    }

    # run experiments
    # if all_args.share_policy:
    #     from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    # else:
    #     from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
    from onpolicy.runner.quartz.quartz_runner import QuartzRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
