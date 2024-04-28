import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self,
                 register_embedding_dimension,      # dimension of register embedding
                 device,                            # device the network is on
                 ):
        super(ValueNetwork, self).__init__()
        hidden_dim1 = 8 * math.ceil(register_embedding_dimension / 3 / 8)
        hidden_dim2 = 8 * math.ceil(register_embedding_dimension / 11 / 8)
        self.linear1 = nn.Linear(register_embedding_dimension, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1, bias=False)
        # self.linear1_initial = nn.Linear(register_embedding_dimension, hidden_dim1)
        # self.linear2_initial = nn.Linear(hidden_dim1, hidden_dim2)
        # self.linear3_initial = nn.Linear(hidden_dim2, 1, bias=False)
        self.to(device)
        # self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.01) * 0.1
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_normal_(self.linear3.weight, gain=gain)
        # nn.init.xavier_normal_(self.linear1_initial.weight, gain=gain)
        # nn.init.zeros_(self.linear1_initial.bias)
        # nn.init.xavier_normal_(self.linear2_initial.weight, gain=gain)
        # nn.init.zeros_(self.linear2_initial.bias)
        # nn.init.xavier_normal_(self.linear3_initial.weight, gain=gain)

    def forward(self,
                register_embedding_batch,       # batched register embedding ([sum #regs, register_embedding_dimension])
                register_count_list,            # number of registers in each device (a list, # = batch size)
                is_initial_phase_batch,         # denote each (a list of bools, # = batch size)
                ):
        # recover the real object from wrapper
        register_embedding_batch = register_embedding_batch.obj
        register_count_list = register_count_list.obj
        is_initial_phase_batch = is_initial_phase_batch.obj

        # cut register embedding batch by shape designated in register_count_batch
        # and pool it into global embedding
        # global_embedding_batch: [batch size, register_embedding_dimension]
        # TODO: we assume #regs is the same for each device here
        assert len(set(register_count_list)) == 1, "we assume #regs is the same for each device"
        reshaped_embedding = register_embedding_batch.reshape([len(register_count_list), register_count_list[0],
                                                               register_embedding_batch.shape[1]])
        global_embedding_batch = torch.sum(reshaped_embedding, dim=1)

        # infer value and value_initial from linear layers (a tower architecture)
        # value_initial: [batch size, 1]
        # value: [batch size, 1]
        # is_initial_phase_tensor = torch.tensor(is_initial_phase_batch, dtype=torch.bool, device=self.device).reshape(-1, 1)
        # _internal_1_initial = F.leaky_relu(self.linear1_initial(global_embedding_batch))
        # _internal_2_initial = F.leaky_relu(self.linear2_initial(_internal_1_initial))
        # value_initial = self.linear3_initial(_internal_2_initial)
        # value_initial_mask = is_initial_phase_tensor.to(torch.float64)

        _internal_1 = F.leaky_relu(self.linear1(global_embedding_batch))
        _internal_2 = F.leaky_relu(self.linear2(_internal_1))
        value = self.linear3(_internal_2)
        # value_mask = (~is_initial_phase_tensor).to(torch.float64)

        # final_value = value_initial * value_initial_mask + value * value_mask
        return value

    # def forward_original(self,
    #                      register_embedding_batch,       # batched register embedding ([sum #regs, register_embedding_dimension])
    #                      register_count_list,            # number of registers in each device (a list, # = batch size)
    #                      is_initial_phase_batch,         # denote each (a list of bools, # = batch size)
    #                      ):
    #     # recover the real object from wrapper
    #     register_embedding_batch = register_embedding_batch.obj
    #     register_count_list = register_count_list.obj
    #     is_initial_phase_batch = is_initial_phase_batch.obj
    #
    #     # cut register embedding batch by shape designated in register_count_batch
    #     # and pool it into global embedding
    #     # global_embedding_batch: [batch size, register_embedding_dimension]
    #     global_embedding_list, _cur_pos = [], 0
    #     for reg_count in register_count_list:
    #         # _cur_reg_embedding: [#regs, register_embedding_dimension]
    #         # _cur_global_embedding: [1, register_embedding_dimension]
    #         _cur_reg_embedding = register_embedding_batch[_cur_pos: _cur_pos + reg_count]
    #         _cur_global_embedding = torch.sum(_cur_reg_embedding, dim=0, keepdim=True)
    #         global_embedding_list.append(_cur_global_embedding)
    #         _cur_pos += reg_count
    #     assert _cur_pos == register_embedding_batch.shape[0]
    #     global_embedding_batch = torch.cat(global_embedding_list, dim=0)
    #
    #     # infer value and value_initial from linear layers (a tower architecture)
    #     # value_initial: [batch size, 1]
    #     # value: [batch size, 1]
    #     # is_initial_phase_tensor = torch.tensor(is_initial_phase_batch, dtype=torch.bool, device=self.device).reshape(-1, 1)
    #     # _internal_1_initial = F.leaky_relu(self.linear1_initial(global_embedding_batch))
    #     # _internal_2_initial = F.leaky_relu(self.linear2_initial(_internal_1_initial))
    #     # value_initial = self.linear3_initial(_internal_2_initial)
    #     # value_initial_mask = is_initial_phase_tensor.to(torch.float64)
    #
    #     _internal_1 = F.leaky_relu(self.linear1(global_embedding_batch))
    #     _internal_2 = F.leaky_relu(self.linear2(_internal_1))
    #     value = self.linear3(_internal_2)
    #     # value_mask = (~is_initial_phase_tensor).to(torch.float64)
    #
    #     # final_value = value_initial * value_initial_mask + value * value_mask
    #     return value


class PolicyNetwork(nn.Module):
    def __init__(self,
                 register_embedding_dimension,  # dimension of register embedding
                 device,                        # device the network is on
                 allow_nop,                     # allow nop (0, 0) in action space (Initial: True, Physical: False)
                 ):
        super(PolicyNetwork, self).__init__()
        self.device = device

        # store parameters
        self.action_embedding_dim = register_embedding_dimension * 2

        # initialize network
        hidden_dim1 = 8 * math.ceil(self.action_embedding_dim / 3 / 8)
        hidden_dim2 = 8 * math.ceil(self.action_embedding_dim / 11 / 8)
        self.linear1 = nn.Linear(self.action_embedding_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, 1, bias=False)

        # initialize embedding for nop
        assert allow_nop
        self.nop_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.action_embedding_dim)

        # send to device
        self.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.01) * 0.05
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)
        nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_normal_(self.linear3.weight, gain=1)

    def forward(self,
                register_embedding_batch,       # batched register embedding
                register_count_list,            # a list of #registers in each device
                action_space_batch,             # a list of action spaces
                ):
        # recover the real object from wrapper
        register_embedding_batch = register_embedding_batch.obj
        register_count_list = register_count_list.obj
        action_space_batch = action_space_batch.obj

        # cut register embedding batch by shape designated in register_count_batch
        # global_embedding_batch: [batch size, register_embedding_dimension]
        reg_partition_list = torch.cumsum(torch.tensor(register_count_list[:-1]), dim=0)
        reg_embedding_list = torch.tensor_split(register_embedding_batch, reg_partition_list)

        # get representation for nop: shape [2, reg representation dimension]
        nop_embedding = F.leaky_relu(self.nop_embedding(torch.tensor([0], device=self.device))).reshape(2, -1)

        # get action prob batch
        action_prob_batch = []
        for reg_embedding, action_space in zip(reg_embedding_list, action_space_batch):
            # reg_embedding: [#regs, register_embedding_dimension]
            # action_space: [(x_1, y_1), (x_2, y_2), ...], \forall x_i < y_i

            # preprocess embedding and action space for nop (nop's embedding is concatenated at the end)
            # now reg_embedding: [#regs + 2, register_embedding_dimension]
            nop_reg_id0, nop_reg_id1 = reg_embedding.shape[0], reg_embedding.shape[0] + 1
            reg_embedding = torch.cat([reg_embedding, nop_embedding], dim=0)
            for idx in range(len(action_space)):
                if action_space[idx][0] == 0 and action_space[idx][1] == 0:
                    action_space[idx] = (nop_reg_id0, nop_reg_id1)

            # construct action embedding
            # action_embedding: [#actions, action_embedding_dim]
            action_space_tensor = torch.tensor(action_space)
            reg_id0_tensor = action_space_tensor[:, 0].squeeze()
            reg_id1_tensor = action_space_tensor[:, 1].squeeze()
            action_embedding = torch.cat([reg_embedding[reg_id0_tensor], reg_embedding[reg_id1_tensor]], dim=1)

            # recover action space (since we should not change input)
            # this is more efficient than deepcopy the input
            for idx in range(len(action_space)):
                if action_space[idx][0] == nop_reg_id0 and action_space[idx][1] == nop_reg_id1:
                    action_space[idx] = (0, 0)

            # infer action logits from action embedding and calculate probability
            _hidden_1 = F.leaky_relu(self.linear1(action_embedding))
            _hidden_2 = F.leaky_relu(self.linear2(_hidden_1))
            action_logits = self.linear3(_hidden_2).squeeze()
            action_prob = F.softmax(action_logits, dim=0)
            action_prob_batch.append(action_prob)

        return action_prob_batch

    # def forward_original(self,
    #                      register_embedding_batch,  # batched register embedding
    #                      register_count_list,  # a list of #registers in each device
    #                      action_space_batch,  # a list of action spaces
    #                      ):
    #     # recover the real object from wrapper
    #     register_embedding_batch = register_embedding_batch.obj
    #     register_count_list = register_count_list.obj
    #     action_space_batch = action_space_batch.obj
    #
    #     # cut register embedding batch by shape designated in register_count_batch
    #     # global_embedding_batch: [batch size, register_embedding_dimension]
    #     reg_embedding_list, _cur_pos = [], 0
    #     for reg_count in register_count_list:
    #         # _cur_reg_embedding: [#regs, register_embedding_dimension]
    #         _cur_reg_embedding = register_embedding_batch[_cur_pos: _cur_pos + reg_count]
    #         reg_embedding_list.append(_cur_reg_embedding)
    #         _cur_pos += reg_count
    #     assert _cur_pos == register_embedding_batch.shape[0]
    #
    #     # get action prob batch
    #     action_prob_batch = []
    #     for reg_embedding, action_space in zip(reg_embedding_list, action_space_batch):
    #         # reg_embedding: [#regs, register_embedding_dimension]
    #         # action_space: [(x_1, y_1), (x_2, y_2), ...], \forall x_i < y_i
    #
    #         # decode action and construct action embedding
    #         # action_embedding: [#actions, action_embedding_dim]
    #         action_embedding_list = []
    #         for action in action_space:
    #             # decode
    #             idx0, idx1 = action[0], action[1]
    #             assert idx0 < idx1 or idx0 == idx1 == 0, "In each action, idx0 must be smaller than idx1"
    #
    #             # parse special case of nop
    #             if idx0 == 0 and idx1 == 0:
    #                 assert self.nop_embedding is not None, "Invalid NOP detected!"
    #                 _cur_action_embedding = self.nop_embedding(torch.tensor([0], device=self.device))
    #                 _cur_action_embedding = F.leaky_relu(_cur_action_embedding)
    #                 _cur_action_embedding = _cur_action_embedding.reshape(1, self.action_embedding_dim)
    #                 action_embedding_list.append(_cur_action_embedding)
    #                 continue
    #
    #             # construct action embedding
    #             # _cur_action_embedding: [1, action_embedding_dim]
    #             _cur_action_embedding = torch.cat([reg_embedding[idx0], reg_embedding[idx1]], dim=0)
    #             _cur_action_embedding = _cur_action_embedding.reshape(1, self.action_embedding_dim)
    #             action_embedding_list.append(_cur_action_embedding)
    #         action_embedding = torch.cat(action_embedding_list, dim=0)
    #
    #         # infer action logits from action embedding and calculate probability
    #         _hidden_1 = F.leaky_relu(self.linear1(action_embedding))
    #         _hidden_2 = F.leaky_relu(self.linear2(_hidden_1))
    #         action_logits = self.linear3(_hidden_2).squeeze()
    #         action_prob = F.softmax(action_logits, dim=0)
    #         action_prob_batch.append(action_prob)
    #
    #     return action_prob_batch
