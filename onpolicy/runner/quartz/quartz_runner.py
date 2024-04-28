import shutil
import time
import wandb
import os
import numpy as np
from itertools import chain
from tqdm import tqdm
import torch
import torch.distributed as dist

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.quartz.quartz_base_runner import QuartzBaseRunner
from onpolicy.runner.quartz.initial_mapping_search import random_search, save_mapping
from onpolicy.utils.quartz_utils import restore_observation, flatten_action
import imageio
import heapq


def _t2n(x):
    return x.detach().cpu().numpy()


class QuartzRunner(QuartzBaseRunner):
    def __init__(self, config):
        super(QuartzRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # final_mapping_dict: a dictionary of final l2p mapping strings to cost
        # backup_best_mapping_dict: stores the best mappings in the last non-empty final_mapping_dict, used when no
        # mapping is found in the current interval
        final_mapping_dict = {}
        backup_final_mapping_dict = None
        backup_best_mapping_dict = None

        for episode in range(episodes):
            # # update initial mapping
            # # TODO: Not used, remove this
            # if episode % self.all_args.search_interval == 0:
            #     if self.all_args.search_type == "random":
            #         if self.rank == 0:
            #             print(f"Start searching for new initial mappings using random search in episode {episode}.")
            #
            #         # prepare file names
            #         mapping_file_path = f"./experiment/{self.all_args.qasm_file_name}/{self.all_args.backend_name}" \
            #                             f"/initial_mapping_dir/mapping_rank{self.rank}.txt"
            #         archive_mapping_file_name = f"./experiment/{self.all_args.qasm_file_name}" \
            #                                     f"/{self.all_args.backend_name}/initial_mapping_dir" \
            #                                     f"/mapping_rank{self.rank}_before{episode}.txt"
            #         model_path = f"./experiment/{self.all_args.qasm_file_name}/{self.all_args.backend_name}" \
            #                      f"/eval_model_dir/model_{last_model_save_episode}.pt"
            #
            #         # random search
            #         new_mapping_dict = random_search(all_args=self.all_args,
            #                                          ddp_rank=self.rank,
            #                                          episode=episode,
            #                                          mapping_file_path=mapping_file_path,
            #                                          model_path=model_path)
            #
            #         # move old mappings and save new ones
            #         os.rename(mapping_file_path, archive_mapping_file_name)
            #         save_mapping(mapping_file_path, new_mapping_dict)
            #
            #     elif self.all_args.search_type == "mcts":
            #         print("MCTS based initial mapping search not implemented")
            #         raise NotImplementedError
            #     elif self.all_args.search_type == "none":
            #         pass
            #     else:
            #         raise NotImplementedError

            # update the mapping every self.all_args.two_way_save_interval epochs in two-way training
            # we allow the rl agent to fine-tune (train value and policy) for sometime before two-way starts
            two_way_saved_mapping_count, two_way_used_mapping_count = None, None
            two_way_best_cost, two_way_worst_cost = None, None
            if (episode + 1) % self.all_args.two_way_save_interval == 0 and not self.all_args.two_way_mode == "none" \
                    and (episode + 1) >= self.all_args.two_way_start_epoch:
                # extract the best mappings (a list of mapping strings) and check that best_mappings is not empty
                # if it's empty, use the backup
                best_mappings = sorted(final_mapping_dict, key=lambda m_str: final_mapping_dict[m_str])[
                                0: self.all_args.two_way_save_count]
                two_way_saved_mapping_count = len(best_mappings)
                if two_way_saved_mapping_count == 0:
                    # if best_mappings is empty, use the backup set
                    print(f"[{self.all_args.two_way_mode} - {self.rank}] Warning: No new mapping found in the current"
                          f" two way interval, continue with the last set of mappings.")
                    assert backup_best_mapping_dict is not None, "Error: Try to use empty backup set!"
                    assert backup_final_mapping_dict is not None, "Error: Try to use empty backup set!"
                    best_mappings = backup_best_mapping_dict.copy()
                    final_mapping_dict = backup_final_mapping_dict.copy()
                two_way_used_mapping_count = len(best_mappings)
                two_way_best_cost = final_mapping_dict[best_mappings[0]]
                two_way_worst_cost = final_mapping_dict[best_mappings[-1]]

                # backup the current mappings in case it is used later
                backup_best_mapping_dict = best_mappings.copy()
                backup_final_mapping_dict = final_mapping_dict.copy()

                # save the mappings into a temp file call mapping_rank_step.txt for the other trainer
                tmp_mapping_file_name = f"./experiment/{self.all_args.reversed_qasm_file_name}" \
                                        f"/{self.all_args.backend_name}/initial_mapping_dir" \
                                        f"/mapping_rank{self.rank}_step{episode + 1}.txt"
                with open(tmp_mapping_file_name, "w") as file:
                    for mapping_str in best_mappings:
                        file.write(f"{mapping_str}\n")
                print(f"Info: Rank {self.rank} saved {self.all_args.two_way_save_count} mappings"
                      f" to {tmp_mapping_file_name}, with best cost {two_way_best_cost} and"
                      f" worst cost {two_way_worst_cost}.")

                # clear the buffer if needed
                if self.all_args.two_way_clear_on_save:
                    print(f"Info: Rank {self.rank} clears the buffer after save (--two_way_clear_on_save=True)!")
                    final_mapping_dict = {}

                # wait for the other side to generate the new mapping for it
                new_mapping_file_name = f"./experiment/{self.all_args.qasm_file_name}" \
                                        f"/{self.all_args.backend_name}/initial_mapping_dir" \
                                        f"/mapping_rank{self.rank}_step{episode + 1}.txt"
                while not os.path.exists(new_mapping_file_name):
                    time.sleep(1)
                print(f"Info: Rank {self.rank} find new mappings from the other side!")

                # save the mapping file in wandb
                shutil.copyfile(src=new_mapping_file_name,
                                dst=str(wandb.run.dir) + f"/mappings/mapping_rank{self.rank}_step{episode + 1}.txt")

                # replace mapping_rank.txt with the new mapping file
                active_mapping_file_name = f"./experiment/{self.all_args.qasm_file_name}" \
                                           f"/{self.all_args.backend_name}/initial_mapping_dir" \
                                           f"/mapping_rank{self.rank}.txt"
                os.remove(active_mapping_file_name)
                shutil.copyfile(src=new_mapping_file_name, dst=active_mapping_file_name)
                assert os.path.exists(active_mapping_file_name), "Mapping file not found!"
                print(f"Info: Rank {self.rank} has successfully updated to new mapping file in step {episode + 1}!")

                # synchronize the ddp trainers
                dist.barrier()

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            if self.rank == 0:
                print(f"Start collecting data for episode {episode}.")
            collect_bar = tqdm(range(self.episode_length)) if self.rank == 0 else range(self.episode_length)
            for step in collect_bar:
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # two-way: save mappings
                # dones: [[false], [false], ...]
                # infos: ([{dict}], [{dict}], ... )
                for idx in range(len(dones)):
                    # save the mapping if an env is finished
                    if dones[idx][0]:
                        # extract data
                        cur_mapping = infos[idx][0]["final_mapping"]
                        cur_cost = infos[idx][0]["cur_total_cost"]
                        assert cur_mapping is not None and not cur_cost == -1

                        # serialize mapping and save
                        mapping_str = ""
                        for logical_idx in range(len(cur_mapping)):
                            mapping_str += f"{cur_mapping[logical_idx]} "
                        final_mapping_dict.update({mapping_str: cur_cost})

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            if self.rank == 0:
                print(f"Start training for episode {episode}.")
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                if self.rank == 0:
                    self.save(episode=episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                if self.rank == 0:
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                          .format(self.all_args.scenario_name, self.algorithm_name, self.experiment_name, episode,
                                  episodes, total_num_steps, self.num_env_steps, int(total_num_steps / (end - start))))

                if self.env_name == "quartz_physical" or self.env_name == "quartz_initial":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        total_cost_list = []
                        best_fidelity_list = []
                        fidelity_of_best_gate_count_circuit_list = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                            if 'best_total_cost' in info[agent_id].keys():
                                total_cost_list.append(info[agent_id]['best_total_cost'])
                            if 'best_fidelity' in info[agent_id].keys():
                                best_fidelity_list.append(info[agent_id]['best_fidelity'])
                            if 'fidelity_of_best_gate_count_circuit' in info[agent_id].keys():
                                fidelity_of_best_gate_count_circuit_list.append(info[agent_id]['fidelity_of_best_gate_count_circuit'])
                        train_infos[agent_id].update({'collect_individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({'collect_best_total_cost': np.min(total_cost_list)})
                        train_infos[agent_id].update({'collect_ln_fidelity_of_best_gate_count_circuit': fidelity_of_best_gate_count_circuit_list[np.argmin(total_cost_list)]})
                        train_infos[agent_id].update({'collect_best_ln_fidelity': np.max(best_fidelity_list)})
                        train_infos[agent_id].update(
                            {"collect_average_episode_rewards": np.mean(
                                self.buffer[agent_id].rewards) * self.episode_length})

                        # two-way's logging
                        if two_way_saved_mapping_count is not None:
                            assert two_way_used_mapping_count is not None
                            assert two_way_best_cost is not None and two_way_worst_cost is not None
                            # two_way_mappings_found: the number of mappings found in the last two-way interval
                            # two_way_mappings_used: the number of mappings used for the following two-way interval
                            # The two values may be different because we may use the backup set when no new mapping
                            # is found in the last two-way interval
                            train_infos[agent_id].update({'two_way_mappings_found': two_way_saved_mapping_count})
                            train_infos[agent_id].update({'two_way_mappings_used': two_way_used_mapping_count})
                            train_infos[agent_id].update({'two_way_best_cost': two_way_best_cost})
                            train_infos[agent_id].update({'two_way_worst_cost': two_way_worst_cost})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                assert False, "In the final version, we should disable eval to save training time."
                if self.rank == 0:
                    print(f"Start evaluation for episode {episode}.")
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            # quartz: restore obs
            circuit_batch, device_edge_list_batch, logical2physical_mapping_batch, \
                physical2logical_mapping_batch, action_space_batch, is_initial_phase_batch = [], [], [], [], [], []
            for obs in self.buffer[agent_id].obs[step]:
                circuit, device_edge_list, logical2physical, physical2logical_mapping, action_space, is_initial_phase \
                    = restore_observation(obs)
                circuit_batch.append(circuit)
                device_edge_list_batch.append(device_edge_list)
                logical2physical_mapping_batch.append(logical2physical)
                physical2logical_mapping_batch.append(physical2logical_mapping)
                action_space_batch.append(action_space)
                is_initial_phase_batch.append(is_initial_phase)

            value, action, action_log_prob \
                = self.trainer[agent_id].policy.get_actions(circuit_batch=circuit_batch,
                                                            device_edge_list_batch=device_edge_list_batch,
                                                            physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                            logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                            action_space_batch=action_space_batch,
                                                            is_initial_phase_batch=is_initial_phase_batch)

            # quartz: reshape
            # make value one dimensional
            value = value.reshape(value.shape[0], 1)
            # make action [(a, b), (c, d) ...] -> np.array(ab, cd, ...)
            action_tmp = [[flatten_action(_action[0], _action[1])] for _action in action]
            action = np.array(action_tmp)
            # make action log prob a numpy array
            action_log_prob = np.array([[_prob.cpu().numpy()] for _prob in action_log_prob])
            # some redundancy
            rnn_state = np.zeros([value.shape[0], self.recurrent_N, self.hidden_size])
            rnn_state_critic = np.zeros([value.shape[0], self.recurrent_N, self.hidden_size])

            # [agents, envs, dim]
            values.append(_t2n(value))
            # rearrange action
            # if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
            #     for i in range(self.envs.action_space[agent_id].shape):
            #         uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
            #         if i == 0:
            #             action_env = uc_action_env
            #         else:
            #             action_env = np.concatenate((action_env, uc_action_env), axis=1)
            # elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
            #     action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            # else:
            #     raise NotImplementedError

            actions.append(action)
            actions_env = list(action)
            action_log_probs.append(action_log_prob)
            rnn_states.append(rnn_state)
            rnn_states_critic.append(rnn_state_critic)

        # [envs, agents, dim]
        # actions_env = []
        # for i in range(self.n_rollout_threads):
        #     one_hot_action_env = []
        #     for temp_action_env in temp_actions_env:
        #         one_hot_action_env.append(temp_action_env[i])
        #     actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #                                      dtype=np.float32)
        # rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #                                             dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # share_obs is not used, so we use a dummy value here
        # share_obs = []
        # for o in obs:
        #     share_obs.append(list(chain(*o)))
        # share_obs = np.array(share_obs)
        share_obs = np.zeros([self.n_rollout_threads, self.all_args.max_obs_length])

        for agent_id in range(self.num_agents):
            # if not self.use_centralized_V:
            #     share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        # eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
        #                            dtype=np.float32)
        # eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # prepare an array for results
        eval_total_cost = np.ones(self.n_eval_rollout_threads, dtype=int) * self.all_args.eval_max_gate_count
        eval_terminate_step = np.ones(self.n_eval_rollout_threads, dtype=int) * self.eval_episode_length
        finished_count = 0

        eval_bar = tqdm(range(self.eval_episode_length)) if self.rank == 0 else range(self.eval_episode_length)
        for _cur_step in eval_bar:
            eval_actions_env = None
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()

                # quartz: reconstruct obs
                circuit_batch, device_edge_list_batch, logical2physical_mapping_batch, \
                    physical2logical_mapping_batch, action_space_batch = [], [], [], [], []
                for obs in np.array(list(eval_obs[:, agent_id])):
                    circuit, device_edge_list, logical2physical, physical2logical_mapping, action_space, \
                        is_initial_phase = restore_observation(obs)
                    circuit_batch.append(circuit)
                    device_edge_list_batch.append(device_edge_list)
                    logical2physical_mapping_batch.append(logical2physical)
                    physical2logical_mapping_batch.append(physical2logical_mapping)
                    action_space_batch.append(action_space)

                eval_action = self.trainer[agent_id].policy.act(circuit_batch=circuit_batch,
                                                                device_edge_list_batch=device_edge_list_batch,
                                                                physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                                logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                                action_space_batch=action_space_batch,
                                                                deterministic=True)
                eval_action = [flatten_action(_action[0], _action[1]) for _action in eval_action]
                eval_actions_env = eval_action

                # rearrange action
                # if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                #     for i in range(self.eval_envs.action_space[agent_id].shape):
                #         eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                #             eval_action[:, i]]
                #         if i == 0:
                #             eval_action_env = eval_uc_action_env
                #         else:
                #             eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                # elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                #     eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                # else:
                #     raise NotImplementedError
                #
                # eval_temp_actions_env.append(eval_action_env)
                # eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            # eval_actions_env = []
            # for i in range(self.n_eval_rollout_threads):
            #     eval_one_hot_action_env = []
            #     for eval_temp_action_env in eval_temp_actions_env:
            #         eval_one_hot_action_env.append(eval_temp_action_env[i])
            #     eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # store evaluation result
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

            # eval_rnn_states[eval_dones == True] = np.zeros(
            #     ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            # eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            # eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        # get best result
        min_total_cost = min(eval_total_cost)
        max_total_cost = max(eval_total_cost)
        avg_total_cost = sum(eval_total_cost) / len(eval_total_cost)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_total_reward = 0
            for idx, valid_step_count in enumerate(eval_terminate_step):
                eval_total_reward += np.sum(eval_episode_rewards[0: valid_step_count + 1, idx, agent_id])
            eval_average_episode_rewards = eval_total_reward / self.n_eval_rollout_threads
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards,
                                     'eval_min_total_cost': min_total_cost,
                                     'eval_max_total_cost': max_total_cost,
                                     'eval_avg_total_cost': avg_total_cost})
            print(f"[rank {self.rank}] eval average episode rewards of agent%i: " % agent_id + str(
                eval_average_episode_rewards))
            print(f"[rank {self.rank}] Best implementation found: {min_total_cost} gates.\n")

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        raise NotImplementedError
