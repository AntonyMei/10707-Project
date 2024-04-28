import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from onpolicy.algorithms.quartz_ppo_dual.algorithm.representation_network import RepresentationNetwork
from onpolicy.algorithms.quartz_ppo_dual.algorithm.actor_critic import ValueNetwork, PolicyNetwork
from onpolicy.algorithms.utils.util import DummyObjWrapper

from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence


class QuartzPPONetwork(nn.Module):
    def __init__(self,
                 # initial register embedding
                 reg_degree_types,                  # number of different register degrees
                 reg_degree_embedding_dim,          # dimension of register feature (degree) embedding
                 # initial qubit embedding
                 gate_is_input_embedding_dim,       # dimension of gate feature (is_input) embedding
                 # graph convolution layers
                 num_gnn_layers,                    # number of graph neural network layers
                 reg_representation_dim,            # dimension of register representation
                 gate_representation_dim,           # dimension of gate representation
                 # device
                 device,                            # device the network is on
                 rank,                              # rank of current model (ddp)
                 # general
                 allow_nop,                         # allow nop (0, 0) in action space
                                                    # (Initial: True, Physical: False)
                 ):
        super(QuartzPPONetwork, self).__init__()
        self.device = device
        self.rank = rank

        # initial model
        self.representation_network = RepresentationNetwork(reg_degree_types=reg_degree_types,
                                                            reg_degree_embedding_dim=reg_degree_embedding_dim,
                                                            gate_is_input_embedding_dim=gate_is_input_embedding_dim,
                                                            num_gnn_layers=num_gnn_layers,
                                                            reg_representation_dim=reg_representation_dim,
                                                            gate_representation_dim=gate_representation_dim,
                                                            device=device)
        self.value_network = ValueNetwork(register_embedding_dimension=reg_representation_dim+gate_representation_dim,
                                          device=device)
        self.policy_network = PolicyNetwork(register_embedding_dimension=reg_representation_dim+gate_representation_dim,
                                            device=device, allow_nop=allow_nop)

        # wrap models with ddp if rank is not None
        # set find_unused_parameters to True to avoid error when there are unused parameters
        if rank is not None:
            self.representation_network = DDP(self.representation_network, device_ids=[rank])
            self.value_network = DDP(self.value_network, device_ids=[rank])
            self.policy_network = DDP(self.policy_network, device_ids=[rank], find_unused_parameters=True)

    def policy_forward(self,
                       circuit_batch,                       # a list of GraphState objects
                       device_edge_list_batch,              # a list of device_edge_list (each has shape=(#edges, 2))
                       physical2logical_mapping_batch,      # a list of physical -> logical mapping (each is a np.array)
                       logical2physical_mapping_batch,      # a list of logical -> physical mapping (each is a np.array)
                       action_space_batch,                  # a list of action spaces (each is a list of (x, y))
                       deterministic=False):
        """
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: selected_action_list: a list of tuples (qubit idx 0, qubit idx 1)
                selected_action_prob_list: a list of log probability of selected action
        """
        # get action probability
        batched_reg_embedding, reg_count_list = \
            self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
                                        device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
                                        physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
                                        logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
        action_prob_batch = self.policy_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                                register_count_list=DummyObjWrapper(reg_count_list),
                                                action_space_batch=DummyObjWrapper(action_space_batch))

        # sample action and return
        selected_action_list, selected_action_prob_list = [], []
        for action_prob, action_space in zip(action_prob_batch, action_space_batch):
            action_dist = Categorical(probs=action_prob)
            selected_action_id = action_dist.mode if deterministic else action_dist.sample()
            selected_action_log_prob = action_dist.log_prob(selected_action_id)
            selected_action_list.append(action_space[selected_action_id])
            selected_action_prob_list.append(selected_action_log_prob)
        return selected_action_list, selected_action_prob_list

    def evaluate_action(self,
                        circuit_batch,                      # a list of GraphState objects
                        device_edge_list_batch,             # a list of device_edge_list (each has shape=(#edges, 2))
                        physical2logical_mapping_batch,     # a list of physical -> logical mapping (each is a np.array)
                        logical2physical_mapping_batch,     # a list of logical -> physical mapping (each is a np.array)
                        action_space_batch,                 # a list of action spaces (each is a list of (x, y))
                        action_id_batch,                    # a list of selected action ids
                        ):
        """
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
                action_id_batch: list of selected actions index
        output: selected_action_prob_list: a list of log probability of selected action
                dist_entropy_list: a list of distribution entropy
        """
        # get action probability
        batched_reg_embedding, reg_count_list = \
            self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
                                        device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
                                        physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
                                        logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
        action_prob_batch = self.policy_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                                register_count_list=DummyObjWrapper(reg_count_list),
                                                action_space_batch=DummyObjWrapper(action_space_batch))

        # return statistics of given action
        selected_action_prob_list, dist_entropy_list = [], []
        for action_prob, action_id in zip(action_prob_batch, action_id_batch):
            action_dist = Categorical(probs=action_prob)
            action_log_prob = action_dist.log_prob(action_id)
            dist_entropy = action_dist.entropy().reshape(1)
            selected_action_prob_list.append(action_log_prob)
            dist_entropy_list.append(dist_entropy)
        return selected_action_prob_list, dist_entropy_list

    def value_forward(self,
                      circuit_batch,                    # a list of GraphState objects
                      device_edge_list_batch,           # a list of device_edge_list (each has shape=(#edges, 2))
                      physical2logical_mapping_batch,   # a list of physical -> logical mapping (each is a np.array)
                      logical2physical_mapping_batch,   # a list of logical -> physical mapping (each is a np.array)
                      is_initial_phase_batch,           # a list of bools, # = batch size
                      ):
        """
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
        output: [batch size, 1]
        """
        # get action probability
        batched_reg_embedding, reg_count_list = \
            self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
                                        device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
                                        physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
                                        logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
        value_batch = self.value_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                         register_count_list=DummyObjWrapper(reg_count_list),
                                         is_initial_phase_batch=DummyObjWrapper(is_initial_phase_batch))
        return value_batch

    def policy_value_forward(self,
                             circuit_batch,                     # a list of GraphState objects
                             device_edge_list_batch,            # a list of device_edge_list (each has shape=(#edges, 2))
                             physical2logical_mapping_batch,    # a list of physical -> logical mapping (each is a np.array)
                             logical2physical_mapping_batch,    # a list of logical -> physical mapping (each is a np.array)
                             action_space_batch,                # a list of action spaces (each is a list of (x, y))
                             is_initial_phase_batch,            # a list of bools, # = batch size
                             ):
        """
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: selected_action_list: a list of tuples (qubit idx 0, qubit idx 1)
                selected_action_prob_list: a list of log probability of selected action
                value batch
        """
        # get action probability
        batched_reg_embedding, reg_count_list = \
            self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
                                        device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
                                        physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
                                        logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
        action_prob_batch = self.policy_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                                register_count_list=DummyObjWrapper(reg_count_list),
                                                action_space_batch=DummyObjWrapper(action_space_batch))

        # value
        value_batch = self.value_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                         register_count_list=DummyObjWrapper(reg_count_list),
                                         is_initial_phase_batch=DummyObjWrapper(is_initial_phase_batch))

        # sample action and return
        selected_action_list, selected_action_prob_list = [], []
        for action_prob, action_space in zip(action_prob_batch, action_space_batch):
            action_dist = Categorical(probs=action_prob)
            selected_action_id = action_dist.sample()
            selected_action_log_prob = action_dist.log_prob(selected_action_id)
            selected_action_list.append(action_space[selected_action_id])
            selected_action_prob_list.append(selected_action_log_prob)

        return selected_action_list, selected_action_prob_list, value_batch

    def evaluate_action_value_forward(self,
                                      circuit_batch,                    # a list of GraphState objects
                                      device_edge_list_batch,           # a list of device_edge_list (each has shape=(#edges, 2))
                                      physical2logical_mapping_batch,   # a list of physical -> logical mapping (each is a np.array)
                                      logical2physical_mapping_batch,   # a list of logical -> physical mapping (each is a np.array)
                                      action_space_batch,               # a list of action spaces (each is a list of (x, y))
                                      action_id_batch,                  # a list of selected action ids
                                      is_initial_phase_batch            # a list of bools, # = batch size
                                      ):
        """
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
                action_id_batch: list of selected actions index
        output: selected_action_log_probs: [#batchsize, 1]
                dist_entropy: a number (as a tensor)
                value_batch: [#batchsize, 1]
        """
        # get action probability
        # batched_reg_embedding: [sum #regs, self.hidden_dimension]
        # reg_count_list: a list that stores #registers for each batch element
        batched_reg_embedding, reg_count_list = \
            self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
                                        device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
                                        physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
                                        logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
        # action_prob_batch: a list of #batch size elements, each element is a probability distribution
        action_prob_batch = self.policy_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                                register_count_list=DummyObjWrapper(reg_count_list),
                                                action_space_batch=DummyObjWrapper(action_space_batch))

        # value
        # value: [batch size, 1]
        value_batch = self.value_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
                                         register_count_list=DummyObjWrapper(reg_count_list),
                                         is_initial_phase_batch=DummyObjWrapper(is_initial_phase_batch))

        # return statistics of given action
        # padded_action_prob_batch: [#batch size, padded action prob length]
        # selected_action_log_probs: [#batch size, 1]
        # dist_entropy: a number as a tensor, shape = []
        padded_action_prob_batch = pad_sequence(action_prob_batch, padding_value=0.0, batch_first=True)
        tensor_action_id_batch = torch.tensor(action_id_batch, device=self.device)
        action_dist_batch = Categorical(probs=padded_action_prob_batch)
        selected_action_log_probs = action_dist_batch.log_prob(tensor_action_id_batch).unsqueeze(dim=1)
        dist_entropy = action_dist_batch.entropy().mean()

        return selected_action_log_probs, dist_entropy, value_batch

    # def evaluate_action_value_forward_original(self,
    #                                            circuit_batch,                    # a list of GraphState objects
    #                                            device_edge_list_batch,           # a list of device_edge_list (each is a list of edges)
    #                                            physical2logical_mapping_batch,   # a list of physical -> logical mapping (each is a map)
    #                                            logical2physical_mapping_batch,   # a list of logical -> physical mapping (each is a map)
    #                                            action_space_batch,               # a list of action spaces (each is a list of (x, y))
    #                                            action_id_batch,                  # a list of selected action ids
    #                                            is_initial_phase_batch            # a list of bools, # = batch size
    #                                            ):
    #     """
    #     input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
    #             logical2physical_mapping_batch: list of observations
    #             action_space_batch: list of decoded action space (see utils.DecodePyActionList)
    #             action_id_batch: list of selected actions index
    #     output: selected_action_prob_list: a list of log probability of selected action
    #             dist_entropy_list: a list of distribution entropy
    #             value batch
    #     """
    #     # get action probability
    #     batched_reg_embedding, reg_count_list = \
    #         self.representation_network(circuit_batch=DummyObjWrapper(circuit_batch),
    #                                     device_edge_list_batch=DummyObjWrapper(device_edge_list_batch),
    #                                     physical2logical_mapping_batch=DummyObjWrapper(physical2logical_mapping_batch),
    #                                     logical2physical_mapping_batch=DummyObjWrapper(logical2physical_mapping_batch))
    #     action_prob_batch = self.policy_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
    #                                             register_count_list=DummyObjWrapper(reg_count_list),
    #                                             action_space_batch=DummyObjWrapper(action_space_batch))
    #
    #     # value
    #     value_batch = self.value_network(register_embedding_batch=DummyObjWrapper(batched_reg_embedding),
    #                                      register_count_list=DummyObjWrapper(reg_count_list),
    #                                      is_initial_phase_batch=DummyObjWrapper(is_initial_phase_batch))
    #
    #     # return statistics of given action
    #     selected_action_prob_list, dist_entropy_list = [], []
    #     for action_prob, action_id in zip(action_prob_batch, action_id_batch):
    #         action_dist = Categorical(probs=action_prob)
    #         action_log_prob = action_dist.log_prob(action_id)
    #         dist_entropy = action_dist.entropy().reshape(1)
    #         selected_action_prob_list.append(action_log_prob)
    #         dist_entropy_list.append(dist_entropy)
    #     return selected_action_prob_list, dist_entropy_list, value_batch
