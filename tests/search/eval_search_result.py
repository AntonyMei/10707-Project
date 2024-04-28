#!/usr/bin/env python
from onpolicy.envs.quartz_physical.quartz_env import SimplePhysicalEnv
import sys
import os
import wandb
import socket
import SMOS
import setproctitle
import time
import numpy as np
import multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.scripts.train.train_quartz import make_eval_env, parse_args
from onpolicy.utils.quartz_utils import restore_observation, flatten_action
from onpolicy.algorithms.quartz_ppo_dual.quartz_ppo_model import QuartzPPOModel as Policy


class QuartzEvaluator:
    def __init__(self, rank, all_args):
        # store input parameters
        self.rank = rank
        self.all_args = all_args
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.eval_episode_length = self.all_args.eval_episodes
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # initialize environments and models
        self.policy = None
        self.eval_envs = None
        self.current_model_step = 0

    def evaluate(self, epoch):
        # initialize
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        self.policy.actor_critic.eval()

        # prepare an array for results
        eval_total_cost = np.ones(self.n_eval_rollout_threads, dtype=int) * self.all_args.eval_max_gate_count
        eval_terminate_step = np.ones(self.n_eval_rollout_threads, dtype=int) * self.eval_episode_length
        finished_count = 0

        # start evaluation
        for _cur_step in range(self.eval_episode_length):
            # quartz: reconstruct obs
            circuit_batch, device_edge_list_batch, logical2physical_mapping_batch, \
                physical2logical_mapping_batch, action_space_batch = [], [], [], [], []
            for obs in np.array(list(eval_obs[:, 0])):
                circuit, device_edge_list, logical2physical, physical2logical_mapping, action_space, is_initial_phase \
                    = restore_observation(obs)
                circuit_batch.append(circuit)
                device_edge_list_batch.append(device_edge_list)
                logical2physical_mapping_batch.append(logical2physical)
                physical2logical_mapping_batch.append(physical2logical_mapping)
                action_space_batch.append(action_space)

            eval_action = self.policy.act(circuit_batch=circuit_batch,
                                          device_edge_list_batch=device_edge_list_batch,
                                          physical2logical_mapping_batch=physical2logical_mapping_batch,
                                          logical2physical_mapping_batch=logical2physical_mapping_batch,
                                          action_space_batch=action_space_batch,
                                          deterministic=False)
            eval_action = [flatten_action(_action[0], _action[1]) for _action in eval_action]
            eval_actions_env = eval_action

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # store evaluation final result if finished
            for idx, zip_pack in enumerate(zip(eval_dones, eval_infos)):
                done, info = zip_pack[0][0], zip_pack[1]
                assert type(done) == np.bool_ or type(done) == bool
                assert "cur_total_cost" in info[0].keys()
                if done and eval_total_cost[idx] == self.all_args.eval_max_gate_count:
                    eval_total_cost[idx] = info[0]['cur_total_cost']
                    eval_terminate_step[idx] = _cur_step
                    finished_count += 1
            if finished_count == self.n_eval_rollout_threads:
                break

        # get summarized result
        min_total_cost = min(eval_total_cost)
        max_total_cost = max(eval_total_cost)
        avg_total_cost = sum(eval_total_cost) / len(eval_total_cost)

        # collect final results
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_total_reward = 0
        for idx, valid_step_count in enumerate(eval_terminate_step):
            eval_total_reward += np.sum(eval_episode_rewards[0: valid_step_count + 1, idx, 0])
        eval_average_episode_rewards = eval_total_reward / self.n_eval_rollout_threads
        eval_train_infos = {'eval_average_episode_rewards': eval_average_episode_rewards,
                            'eval_min_total_cost': min_total_cost,
                            'eval_max_total_cost': max_total_cost,
                            'eval_avg_total_cost': avg_total_cost}
        print(f"[rank {self.rank}] model epoch {epoch}: eval average episode rewards = {eval_average_episode_rewards},"
              f" best implementation found = {min_total_cost}.")

        # return final result
        return eval_train_infos


def main(args):
    # parse arguments
    parser = get_config()
    all_args = parse_args(args, parser)
    assert all_args.algorithm_name == "quartz_ppo"

    # start smos server and client for message passing
    smos_server = SMOS.Server(SMOS.ConnectionDescriptor("localhost", 3001, b'antony'))
    smos_server.start()

    # create client and queue object
    # [eval rank, model step (epoch in training), results]
    smos_client = SMOS.Client(SMOS.ConnectionDescriptor("localhost", 3001, b'antony'))
    smos_client.create_object(name="result_queue", max_capacity=32, track_count=3, block_size=[128, 128, 1024 ** 2])

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/eval_results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # start wandb
    wandb_logger = wandb.init(config=all_args,
                              project=all_args.env_name,
                              entity=all_args.user_name,
                              notes=socket.gethostname(),
                              name=str(all_args.algorithm_name) + "_" +
                                   str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                              group=all_args.scenario_name,
                              dir=str(run_dir),
                              job_type="evaluation",
                              reinit=True,
                              mode="offline")

    # start evaluation processes
    ctx = mp.get_context("spawn")
    eval_worker_group = []
    for eval_idx in range(all_args.world_size):
        eval_worker = ctx.Process(target=start_evaluation, args=(eval_idx, all_args))
        eval_worker.start()
        eval_worker_group.append(eval_worker)

    # start collecting results and log to wandb
    finished_evaluations = 0
    try:
        while True:
            # read result packet from SMOS
            status, handle, data = smos_client.pop_from_object(name="result_queue")
            if not status == SMOS.SMOS_SUCCESS:
                time.sleep(10)
                continue

            # log to terminal and wandb
            eval_rank, model_step, infos = data[0], data[1], data[2]
            print(f"[master] received eval results from rank {eval_rank} using model from epoch {model_step}.")
            wandb.log({"eval_epoch": model_step}, step=finished_evaluations)
            for k, v in infos.items():
                wandb.log({k: v}, step=finished_evaluations)
            finished_evaluations += 1

            # clean up
            smos_client.free_handle(handle)

    except KeyboardInterrupt:
        print("[master] Cleaning up evaluation processes.")
        wandb_logger.finish()
        smos_server.stop()
        for eval_worker in eval_worker_group:
            eval_worker.terminate()
        print("All workers have been stopped.")


def start_evaluation(rank, all_args):
    # determine whether we allow nop
    # if all_args.env_name == "quartz_initial":
    #     allow_nop = True
    # elif all_args.env_name == "quartz_physical":
    #     allow_nop = False
    # else:
    #     raise NotImplementedError

    # initialize evaluator and smos client
    evaluator = QuartzEvaluator(rank=rank, all_args=all_args)
    evaluator.eval_envs = make_eval_env(all_args=all_args, ddp_rank=rank)
    evaluator.policy = Policy(reg_degree_types=all_args.reg_degree_types,
                              reg_degree_embedding_dim=all_args.reg_degree_embedding_dim,
                              gate_is_input_embedding_dim=all_args.gate_is_input_embedding_dim,
                              num_gnn_layers=all_args.num_gnn_layers,
                              reg_representation_dim=all_args.reg_representation_dim,
                              gate_representation_dim=all_args.gate_representation_dim,
                              # optimization process
                              device=torch.device(f"cuda:{rank}"),
                              rank=None,  # set to None so that DDP is not enabled for evaluation
                              lr=all_args.lr,
                              allow_nop=all_args.allow_nop_in_initial,
                              opti_eps=all_args.opti_eps,
                              weight_decay=all_args.weight_decay)
    smos_client = SMOS.Client(connection=SMOS.ConnectionDescriptor(ip="localhost", port=3001, authkey=b"antony"))
    print(f"[evaluator rank {rank}] Initialized!")

    # evaluate
    while True:
        # # determine which model is latest model
        # filename_list = os.listdir(f"./experiment/eval_model_dir/{all_args.qasm_file_name}/{all_args.backend_name}")
        # if len(filename_list) == 0:
        #     time.sleep(10)
        #     continue
        # latest_model_name = None
        # latest_epoch = -1
        # for filename in filename_list:
        #     cur_model_epoch = int(filename.split(".")[0].split("_")[1])
        #     if cur_model_epoch > latest_epoch:
        #         latest_model_name = filename
        #         latest_epoch = cur_model_epoch

        # load latest model (stripe "ddp." prefix from state dict)
        latest_epoch = 2400
        raw_state_dict = torch.load(f"../../onpolicy/scripts/download/model_-3reward.pt")
        state_dict = {'.'.join([k.split('.')[0]] + k.split('.')[2:]): v.cpu() for k, v in raw_state_dict.items()}
        evaluator.policy.actor_critic.load_state_dict(state_dict=state_dict)

        # evaluate
        result = evaluator.evaluate(epoch=latest_epoch)

        # push result into smos queue
        status, _ = smos_client.push_to_object(name="result_queue", data=[rank, latest_epoch, result])
        assert status == SMOS.SMOS_SUCCESS


if __name__ == '__main__':
    main(sys.argv[1:])
