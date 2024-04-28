import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule
from onpolicy.utils.quartz_utils import restore_observation, get_latest_checkpoint_id


def _t2n(x):
    return x.detach().cpu().numpy()


class QuartzBaseRunner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # ddp
        self.rank = config['rank']

        # parameters
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

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from onpolicy.algorithms.quartz_ppo_dual.quartz_ppo import QuartzPPO as TrainAlgo
        from onpolicy.algorithms.quartz_ppo_dual.quartz_ppo_model import QuartzPPOModel as Policy

        # quartz: check --use_centralized_V
        assert not self.use_centralized_V, "add --use_centralized_V to disable it"

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                self.envs.observation_space[agent_id]
            # determine whether we allow nop
            # if self.all_args.env_name == "quartz_initial":
            #     allow_nop = True
            # elif self.all_args.env_name == "quartz_physical":
            #     allow_nop = False
            # else:
            #     raise NotImplementedError
            # policy network
            po = Policy(reg_degree_types=self.all_args.reg_degree_types,
                        reg_degree_embedding_dim=self.all_args.reg_degree_embedding_dim,
                        gate_is_input_embedding_dim=self.all_args.gate_is_input_embedding_dim,
                        num_gnn_layers=self.all_args.num_gnn_layers,
                        reg_representation_dim=self.all_args.reg_representation_dim,
                        gate_representation_dim=self.all_args.gate_representation_dim,
                        # optimization process
                        device=self.device,
                        rank=self.rank,
                        allow_nop=self.all_args.allow_nop_in_initial,
                        lr=self.all_args.lr,
                        opti_eps=self.all_args.opti_eps,
                        weight_decay=self.all_args.weight_decay)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        # # load model if we are in stage 2 ("finetune")
        # # TODO: Obsolete, remove this.
        # latest_epoch = None
        # if self.all_args.search_type == "mcts" or self.all_args.search_type == "random":
        #     print(f"Info: search type is set to {self.all_args.search_type}, continue from stage 1.")
        #     latest_epoch = get_latest_checkpoint_id(f"./experiment/{self.all_args.qasm_file_name}"
        #                                             f"/{self.all_args.backend_name}/eval_model_dir")
        #     # load model
        #     model_state_dict = torch.load(f"./experiment/{self.all_args.qasm_file_name}"
        #                                   f"/{self.all_args.backend_name}"
        #                                   f"/eval_model_dir/model_{latest_epoch}.pt")
        #     self.policy[0].actor_critic.load_state_dict(model_state_dict)

        # load model if we are training in two-way mode (fine-tuning)
        # Note: If we pretrain the whole network, vnorm also need to be restored. (see below)
        # determine the model dir
        if self.all_args.backend_name == "IBM_Q65_HUMMINGBIRD":
            _model_prefix = "IBM_Q65_Hummingbird"
        elif self.all_args.backend_name == "IBM_Q27_FALCON":
            _model_prefix = "IBM_Q27_Falcon"
        else:
            raise NotImplementedError
        # load the model
        if self.all_args.two_way_mode == "forward" or self.all_args.two_way_mode == "backward":
            print(f"Info: Running two-way training, mode={self.all_args.two_way_mode},"
                  f" device={self.all_args.backend_name}.")
            # load model
            if self.all_args.pretrain_mode == "representation":
                # we need to exclude value & policy network parameters from the pretrained model
                model_file_name = f"./pretrained_models/{_model_prefix}_model.pt"
                model_state_dict = torch.load(model_file_name)
                representation_network_state_dict = {}
                for k, v in model_state_dict.items():
                    # ignore value and policy network parameters
                    if k.startswith("policy") or k.startswith("value"):
                        continue

                    # rename representation network parameters
                    if k.startswith("representation_network."):
                        name = k[23:]  # remove `representation_network.`
                    else:
                        name = k
                    representation_network_state_dict[name] = v
                self.policy[0].actor_critic.representation_network.load_state_dict(representation_network_state_dict)
                print(f"Info: --pretrain_mode=representation, Representation network is successfully "
                      f"restored from {model_file_name}!")
            elif self.all_args.pretrain_mode == "full":
                model_file_name = f"./pretrained_models/{_model_prefix}_model.pt"
                model_state_dict = torch.load(model_file_name)
                self.policy[0].actor_critic.load_state_dict(model_state_dict)
                print(f"Info: --pretrain_mode=full, PPO network is successfully restored from {model_file_name}!")
            else:
                raise NotImplementedError

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device, rank=self.rank)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

        # # load ValueNorm parameters if we are in stage 2 ("finetune")
        # # TODO: Obsolete, remove this
        # if (self.all_args.search_type == "mcts" or self.all_args.search_type == "random") and \
        #         self.all_args.use_valuenorm:
        #     # load vnorm parameters
        #     vnorm_dict = torch.load(f"./experiment/{self.all_args.qasm_file_name}"
        #                             f"/{self.all_args.backend_name}"
        #                             f"/eval_model_dir/vnorm_param_{latest_epoch}.pt")
        #     self.trainer[0].value_normalizer.set_parameters(running_mean=vnorm_dict["manual_vnorm_rm"],
        #                                                     running_mean_sq=vnorm_dict["manual_vnorm_rm_sq"],
        #                                                     debiasing_term=vnorm_dict["manual_vnorm_db"])

        # Since we pretrain the whole network, vnorm also need to be restored.
        if (self.all_args.two_way_mode == "forward" or self.all_args.two_way_mode == "backward") and \
                self.all_args.use_valuenorm and self.all_args.pretrain_mode == "full":
            # load vnorm parameters
            vnorm_file_name = f"./pretrained_models/{_model_prefix}_vnorm_param.pt"
            vnorm_dict = torch.load(vnorm_file_name)
            self.trainer[0].value_normalizer.set_parameters(running_mean=vnorm_dict["manual_vnorm_rm"],
                                                            running_mean_sq=vnorm_dict["manual_vnorm_rm_sq"],
                                                            debiasing_term=vnorm_dict["manual_vnorm_db"])
            print(f"Info: --pretrain_mode=full, Vnorm parameters are successfully restored from {vnorm_file_name}!")

        # synchronize here
        dist.barrier()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # quartz: reconstruct obs
            circuit_batch, device_edge_list_batch = [], []
            physical2logical_mapping_batch, logical2physical_mapping_batch = [], []
            is_initial_phase_batch = []
            for obs in self.buffer[agent_id].obs[-1]:
                circuit, device_edge_list, logical2physical, physical2logical_mapping, _, is_initial_phase \
                    = restore_observation(obs)
                circuit_batch.append(circuit)
                device_edge_list_batch.append(device_edge_list)
                logical2physical_mapping_batch.append(logical2physical)
                physical2logical_mapping_batch.append(physical2logical_mapping)
                is_initial_phase_batch.append(is_initial_phase)
            next_value = self.trainer[agent_id].policy.get_values(circuit_batch=circuit_batch,
                                                                  device_edge_list_batch=device_edge_list_batch,
                                                                  physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                                  logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                                  is_initial_phase_batch=is_initial_phase_batch)
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, episode):
        for agent_id in range(self.num_agents):
            # save model
            policy_actor_critic = self.trainer[agent_id].policy.actor_critic
            torch.save(policy_actor_critic.state_dict(),
                       str(self.save_dir) + "/latest_model/actor_critic_" + str(agent_id) + ".pt")
            torch.save(policy_actor_critic.representation_network.state_dict(),
                       str(self.save_dir) + "/latest_model/representation_" + str(agent_id) + ".pt")
            torch.save(policy_actor_critic.state_dict(), f"./experiment/{self.all_args.qasm_file_name}"
                                                         f"/{self.all_args.backend_name}"
                                                         f"/eval_model_dir/model_{episode}.pt")

            # some additional parameters that need to be saved
            if self.all_args.use_valuenorm:
                vnorm_param = {"manual_vnorm_rm": self.trainer[0].value_normalizer.running_mean,
                               "manual_vnorm_rm_sq": self.trainer[0].value_normalizer.running_mean_sq,
                               "manual_vnorm_db": self.trainer[0].value_normalizer.debiasing_term}
                torch.save(vnorm_param, str(self.save_dir) + "/latest_model/vnorm_param" + str(agent_id) + ".pt")
                torch.save(vnorm_param, f"./experiment/{self.all_args.qasm_file_name}"
                                        f"/{self.all_args.backend_name}/eval_model_dir/vnorm_param_{episode}.pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            # load model
            print(f"Info: Load model from {str(self.model_dir) + '/actor_critic' + str(agent_id) + '.pt'}")
            policy_actor_critic_state_dict = torch.load(str(self.model_dir) + '/actor_critic' + str(agent_id) + '.pt')
            self.policy[agent_id].actor_critic.load_state_dict(policy_actor_critic_state_dict)

            # load additional parameters if needed
            if self.all_args.use_valuenorm:
                print(f"Info: parameters of ValueNorm also recovered")
                vnorm_dict = torch.load(str(self.save_dir) + "/vnorm_param" + str(agent_id) + ".pt")
                self.trainer[0].value_normalizer.set_parameters(running_mean=vnorm_dict["manual_vnorm_rm"],
                                                                running_mean_sq=vnorm_dict["manual_vnorm_rm_sq"],
                                                                debiasing_term=vnorm_dict["manual_vnorm_db"])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
