from typing import List
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dgl.nn import SAGEConv
from onpolicy.utils.quartz_utils import (
    device_edges_2_dgl_batch, graph_state_2_dgl_batch,
    graph_state_2_dgl, device_edges_2_dgl, invert_permutation
)
from onpolicy.algorithms.quartz_ppo_dual.algorithm.circuit_conv import CircuitGraphConv


class RepresentationNetwork(nn.Module):
    def __init__(self,
                 # initial register embedding
                 reg_degree_types,  # number of different register degrees
                 reg_degree_embedding_dim,  # dimension of register feature (degree) embedding
                 # initial qubit embedding
                 gate_is_input_embedding_dim,  # dimension of gate feature (is_input) embedding
                 # graph convolution layers
                 num_gnn_layers,  # number of graph neural network layers
                 reg_representation_dim,  # dimension of register representation
                 gate_representation_dim,  # dimension of gate representation
                 # device
                 device,  # device the network is on
                 ):
        super(RepresentationNetwork, self).__init__()
        # store some parameters
        self.reg_embedding_dim = reg_degree_embedding_dim
        self.gate_embedding_dim = gate_is_input_embedding_dim
        self.num_gnn_layers = num_gnn_layers
        self.reg_representation_dim = reg_representation_dim
        self.gate_representation_dim = gate_representation_dim
        self.concat_embedding_dim = reg_degree_embedding_dim + gate_is_input_embedding_dim
        self.hidden_dimension = reg_representation_dim + gate_representation_dim
        self.device = device

        # register and qubit embedding network
        self.reg_embedding_network = nn.Embedding(num_embeddings=reg_degree_types,
                                                  embedding_dim=reg_degree_embedding_dim)
        self.gate_embedding_network = nn.Embedding(num_embeddings=2,  # True or False
                                                   embedding_dim=gate_is_input_embedding_dim)

        # graph neural network layers
        assert num_gnn_layers > 1, f"Too few GNN layers: {num_gnn_layers}."
        # register GNN and gate GNN (this order can make ddp faster)
        self.reg_conv_layers = nn.ModuleList([SAGEConv(in_feats=self.concat_embedding_dim,
                                                       out_feats=reg_representation_dim,
                                                       aggregator_type="mean", activation=F.leaky_relu)])
        self.gate_conv_layers = nn.ModuleList([CircuitGraphConv(gate_embedding_dim=gate_is_input_embedding_dim,
                                                                reg_embedding_dim=reg_degree_embedding_dim,
                                                                new_gate_embedding_dim=gate_representation_dim)])
        for _ in range(num_gnn_layers - 1):
            _cur_reg_conv_layer = SAGEConv(in_feats=self.hidden_dimension,
                                           out_feats=reg_representation_dim,
                                           aggregator_type="mean", activation=F.leaky_relu)
            _cur_gate_conv_layer = CircuitGraphConv(gate_embedding_dim=gate_representation_dim,
                                                    reg_embedding_dim=reg_representation_dim,
                                                    new_gate_embedding_dim=gate_representation_dim)
            self.reg_conv_layers.append(_cur_reg_conv_layer)
            self.gate_conv_layers.append(_cur_gate_conv_layer)

        # move model to target device
        self.to(device)

    def reset_parameters(self):
        gain = nn.init.calculate_gain("linear")
        torch.nn.init.xavier_normal_(self.reg_embedding_network.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.gate_embedding_network.weight, gain=gain)

    def forward(
            self,
            circuit_batch,                      # a list of PyGraphState objects
            device_edge_list_batch,             # a list of device_edge_list (each has shape=(#edges, 2))
            physical2logical_mapping_batch,     # a list of physical -> logical mapping (each is a np.array)
            logical2physical_mapping_batch,     # a list of logical -> physical mapping (each is a np.array)
    ):
        # recover the real object from wrapper
        circuit_batch = circuit_batch.obj
        device_edge_list_batch = device_edge_list_batch.obj
        physical2logical_mapping_batch = physical2logical_mapping_batch.obj
        logical2physical_mapping_batch = logical2physical_mapping_batch.obj

        # check input length
        assert len(circuit_batch) == len(device_edge_list_batch) == \
               len(physical2logical_mapping_batch) == len(logical2physical_mapping_batch)
        batch_size = len(circuit_batch)

        # STEP 1: generate batched dgl graph for device and circuit
        # TODO: we assume #regs is the same for each device here
        batched_device_dgl = device_edges_2_dgl_batch(device_edge_list_batch, device=self.device, dummy=True)
        batched_circuit_dgl, num_qubits_list, edge_reg_indices_list = \
            graph_state_2_dgl_batch(circuit_batch, device=self.device)
        num_regs_list = list(map(len, physical2logical_mapping_batch))
        batch_tot_num_regs = sum(num_regs_list)
        num_gates_list: List[int] = [circ.number_of_nodes for circ in circuit_batch]

        # STEP 2: get initial features from embedding network
        # batched_reg_embeddings: [sum #regs, reg_degree_embedding_dim]
        # batched_gate_embeddings: [sum #gates, gate_is_input_embedding_dim]
        batched_reg_embeddings = self.reg_embedding_network(batched_device_dgl.ndata["degree"])
        batched_gate_embeddings = self.gate_embedding_network(batched_circuit_dgl.ndata["is_input"])

        # STEP 3: device GNN layers and circuit GNN layers
        batched_matched_qubit_embeddings: torch.Tensor  # [sum #regs, self.hidden_dimension]
        for layer_idx in range(self.num_gnn_layers + 1):
            # STEP 3.2: prepare input for current layer's device GNN (reorder and concat)
            batched_matched_qubit_embeddings = torch.zeros(
                batch_tot_num_regs, batched_gate_embeddings.shape[-1], device=self.device
            )

            _reg_indices_in_batch = []
            _qubit_indices_in_batch = []
            _edge_reg_indices_in_batch = []
            _reg_idx_base = 0
            _qubit_idx_base = 0
            for i_batch, circuit in enumerate(circuit_batch):
                num_qubits = num_qubits_list[i_batch]
                qubit_physical_idx = torch.LongTensor(circuit.input_physical_idx[:num_qubits])
                _reg_indices_in_batch.append(qubit_physical_idx + _reg_idx_base)
                _qubit_indices_in_batch.append(torch.arange(qubit_physical_idx.shape[0]) + _qubit_idx_base)
                _edge_reg_indices_in_batch.append(edge_reg_indices_list[i_batch].long() + _reg_idx_base)
                _reg_idx_base += num_regs_list[i_batch]
                _qubit_idx_base += num_gates_list[i_batch]
            # end for
            _reg_indices_in_batch = torch.cat(_reg_indices_in_batch)
            _qubit_indices_in_batch = torch.cat(_qubit_indices_in_batch)
            _edge_reg_indices_in_batch = torch.cat(_edge_reg_indices_in_batch)
            batched_matched_qubit_embeddings[_reg_indices_in_batch] = \
                batched_gate_embeddings[_qubit_indices_in_batch]
            # concatenate with device embeddings
            # batched_reg_qubit_embeddings: loop 0: [#regs, self.concat_embedding_dim]
            #                       loop >= 1: [#regs, self.hidden_dimension]
            batched_reg_qubit_embeddings = torch.cat([
                batched_reg_embeddings, batched_matched_qubit_embeddings
            ], dim=1)

            # in the final loop, we only need to concatenate register embeddings
            # as return value of representation network
            if layer_idx == self.num_gnn_layers:
                break

            # STEP 3.3: prepare input for current layer's circuit GNN (use reg embedding as feature)
            # select reg embedding for each edge based on edge physical idx
            batched_edge_embeddings: torch.Tensor  # loop 0: [#edges, reg_degree_embedding_dim]
            # loop >= 1: [#edges, reg_representation_dim]
            batched_edge_embeddings = batched_reg_embeddings[_edge_reg_indices_in_batch]

            # STEP 3.4: apply device gnn and circuit gnn on embeddings
            # new_batched_reg_embedding: [sum #regs, reg_representation_dim]
            # new_batched_gate_embedding: [sum #gates, gate_representation_dim]
            new_batched_reg_embedding = self.reg_conv_layers[layer_idx](
                graph=batched_device_dgl,
                feat=batched_reg_qubit_embeddings
            )
            new_batched_gate_embedding = self.gate_conv_layers[layer_idx](
                g=batched_circuit_dgl,
                input_gate_embedding=batched_gate_embeddings,
                edge_reg_embedding=batched_edge_embeddings
            )
            batched_reg_embeddings = new_batched_reg_embedding
            batched_gate_embeddings = new_batched_gate_embedding

        # concatenate final reg representation and qubit representation
        # return value shape: [sum #regs, self.hidden_dimension]
        # num_regs_list: a list that stores #registers for each batch element
        return batched_reg_qubit_embeddings, num_regs_list

    # # Below is our old implementation of DGNN, which is really slow
    # def check_forward(self, **kwargs):
    #     return self._forward_batch(**kwargs)
    #
    #     naive = self._forward_naive(**kwargs)
    #     batch = self._forward_batch(**kwargs)
    #     # embed()
    #     assert torch.all(naive[0] == batch[0]) and naive[1] == batch[1]
    #     return batch
    #
    # def _forward_naive(self,
    #                    circuit_batch,  # a list of PyGraphState objects
    #                    device_edge_list_batch,  # a list of device_edge_list (each is a list of edges)
    #                    physical2logical_mapping_batch,  # a list of physical -> logical mapping (each is a map)
    #                    logical2physical_mapping_batch,  # a list of logical -> physical mapping (each is a map)
    #                    ):
    #     # check input length
    #     assert len(circuit_batch) == len(device_edge_list_batch) == len(physical2logical_mapping_batch) \
    #            == len(logical2physical_mapping_batch)
    #     batch_size = len(circuit_batch)
    #
    #     # STEP 1: generate batched dgl graph for device and circuit
    #     # device dgl
    #     device_dgl_list, reg_count_list = [], []
    #     for device_edge_list in device_edge_list_batch:
    #         device_dgl = device_edges_2_dgl(device_edges_list=device_edge_list, device=self.device)
    #         device_dgl_list.append(device_dgl)
    #         reg_count_list.append(device_dgl.number_of_nodes())
    #     batched_device_dgl = dgl.batch(device_dgl_list)
    #     # circuit dgl
    #     circuit_dgl_list, gate_count_list, qubit_count_list = [], [], []
    #     edge_physical_idx_list = []
    #     for circuit in circuit_batch:
    #         circuit_dgl = graph_state_2_dgl(graph_state=circuit, device=self.device)
    #         circuit_dgl_list.append(circuit_dgl)
    #         gate_count_list.append(len(circuit.is_input))
    #         qubit_count_list.append(sum(circuit.is_input))
    #         edge_physical_idx_list.append(circuit_dgl.edata["physical_idx"])
    #     batched_circuit_dgl = dgl.batch(circuit_dgl_list)
    #
    #     # STEP 2: get initial features from embedding network
    #     # reg_embeddings: [sum #regs, reg_degree_embedding_dim]
    #     # gate_embeddings: [sum #gates, gate_is_input_embedding_dim]
    #     batched_reg_embeddings = self.reg_embedding_network(batched_device_dgl.ndata["degree"])
    #     batched_gate_embeddings = self.gate_embedding_network(batched_circuit_dgl.ndata["is_input"])
    #
    #     # STEP 3: device GNN layers and circuit GNN layers
    #     # final_reg_embedding: [sum #regs, self.hidden_dimension]
    #     final_reg_embedding = None
    #     for layer_idx in range(self.num_gnn_layers + 1):
    #         # STEP 3.1: split previous embeddings
    #         # reg_embedding_list: a list, one element for each input device in batch
    #         #                     loop 0: [#regs, reg_degree_embedding_dim]
    #         #                     loop >= 1: [#regs, reg_representation_dim]
    #         # gate_embedding_list: a list, one element for each input circuit in batch
    #         #                      loop 0: [#gates, gate_is_input_embedding_dim]
    #         #                      loop >= 1: [#gates, gate_representation_dim]
    #         reg_embedding_list, gate_embedding_list = [], []
    #         _cur_reg_pos, _cur_gate_pos = 0, 0
    #         for reg_count, gate_count in zip(reg_count_list, gate_count_list):
    #             reg_embedding_list.append(batched_reg_embeddings[_cur_reg_pos: _cur_reg_pos + reg_count])
    #             gate_embedding_list.append(batched_gate_embeddings[_cur_gate_pos: _cur_gate_pos + gate_count])
    #             _cur_reg_pos += reg_count
    #             _cur_gate_pos += gate_count
    #         assert _cur_reg_pos == batched_reg_embeddings.shape[0]
    #         assert _cur_gate_pos == batched_gate_embeddings.shape[0]
    #
    #         # STEP 3.2: prepare input for current layer's device GNN (reorder and concat)
    #         # reorder based on physical mapping to get input for current layer's device gnn
    #         concat_reg_embedding_list = []
    #         for state_idx in range(batch_size):
    #             # gather what we need for reordering
    #             # reg_embedding: loop 0: [#regs, reg_degree_embedding_dim]
    #             #                loop >= 1: [#regs, reg_representation_dim]
    #             # qubit_embedding: loop 0: [#qubits, gate_is_input_embedding_dim]
    #             #                  loop >= 1: [#qubits, gate_representation_dim]
    #             # raw input and statistics
    #             circuit = circuit_batch[state_idx]
    #             physical2logical_mapping = physical2logical_mapping_batch[state_idx]
    #             num_qubits = qubit_count_list[state_idx]
    #             num_registers = len(physical2logical_mapping)
    #             # embedding
    #             reg_embedding = reg_embedding_list[state_idx]
    #             qubit_embedding = gate_embedding_list[state_idx][:num_qubits]
    #
    #             # get padding length, which is equal to logical embedding dimension
    #             physical_padding_length = qubit_embedding.shape[1]
    #
    #             # append empty vector as qubit_embedding for unused registers
    #             # qubit_embedding: loop 0: [#regs, gate_is_input_embedding_dim]
    #             #                  loop >= 1: [#regs, gate_representation_dim]
    #             _qubit_embedding_padding = torch.zeros(num_registers - num_qubits, physical_padding_length,
    #                                                    device=self.device)
    #             qubit_embedding = torch.cat([qubit_embedding, _qubit_embedding_padding], dim=0)
    #
    #             # reorder qubit representation
    #             # This inversion is necessary because logical id may not be continuous, so we can
    #             # not use logical -> physical mapping directly
    #             # sorted_qubit_embedding: loop 0: [#regs, gate_is_input_embedding_dim]
    #             #                         loop >= 1: [#regs, gate_representation_dim]
    #             qubit_physical_idx = list(circuit.input_physical_idx[:num_qubits])
    #             for i in range(num_registers):
    #                 if i not in qubit_physical_idx:
    #                     qubit_physical_idx.append(i)
    #             inverted_physical_idx = np.array(invert_permutation(qubit_physical_idx))
    #             gather_indices = [inverted_physical_idx for _ in range(physical_padding_length)]
    #             gather_indices = np.array(gather_indices)
    #             gather_indices = torch.tensor(gather_indices, device=self.device).transpose(0, 1)
    #             sorted_qubit_embedding = torch.gather(input=qubit_embedding, dim=0, index=gather_indices)
    #
    #             # concatenate with device embedding
    #             # concat_reg_embedding: loop 0: [#regs, self.concat_embedding_dim]
    #             #                       loop >= 1: [#regs, self.hidden_dimension]
    #             concat_reg_embedding = torch.cat([reg_embedding, sorted_qubit_embedding], dim=1)
    #             concat_reg_embedding_list.append(concat_reg_embedding)
    #
    #         # concatenate concat_reg_embedding_list into a batched tensor, this is the
    #         # tensor passed into device GNN layer
    #         # batched_concat_reg_embedding: loop 0: [sum #regs, self.concat_embedding_dim]
    #         #                               loop >= 1: [sum #regs, self.hidden_dimension]
    #         batched_concat_reg_embedding = torch.cat(concat_reg_embedding_list, dim=0)
    #
    #         # in the final loop, we only need to concatenate register embeddings as return
    #         # value of representation network
    #         if layer_idx == self.num_gnn_layers:
    #             final_reg_embedding = batched_concat_reg_embedding
    #             break
    #
    #         # STEP 3.3: prepare input for current layer's circuit GNN (use reg embedding as feature)
    #         # select reg embedding for each edge based on edge physical idx
    #         edge_reg_embedding_list = []
    #         for state_idx in range(batch_size):
    #             # get reg embedding & each edge's physical idx
    #             # reg_embedding: loop 0: [#regs, reg_degree_embedding_dim]
    #             #                loop >= 1: [#regs, reg_representation_dim]
    #             # edge_physical_idx: [#edges, 1]
    #             reg_embedding = reg_embedding_list[state_idx]
    #             edge_physical_idx = edge_physical_idx_list[state_idx].unsqueeze(1)
    #
    #             # prepare gather indices
    #             # gather_indices: loop 0: [#edges, reg_degree_embedding_dim]
    #             #                 loop >= 1: [#edges, reg_representation_dim]
    #             reg_embedding_dim = reg_embedding.shape[1]
    #             _indices_list = [edge_physical_idx for _ in range(reg_embedding_dim)]
    #             gather_indices = torch.cat(_indices_list, dim=1).type(torch.int64)
    #
    #             # gather reg embedding as edge feature
    #             # edge_reg_embedding: loop 0: [#edges, reg_degree_embedding_dim]
    #             #                     loop >= 1: [#edges, reg_representation_dim]
    #             edge_reg_embedding = torch.gather(input=reg_embedding, dim=0, index=gather_indices)
    #             edge_reg_embedding_list.append(edge_reg_embedding)
    #
    #         # concat edge reg embeddings into a batch
    #         # edge_reg_embedding_batch: loop 0: [sum #edges, reg_degree_embedding_dim]
    #         #                           loop >= 1: [sum #edges, reg_representation_dim]
    #         edge_reg_embedding_batch = torch.cat(edge_reg_embedding_list, dim=0)
    #
    #         # STEP 3.4: apply device gnn and circuit gnn on embeddings
    #         # new_batched_reg_embedding: [sum #regs, reg_representation_dim]
    #         # new_batched_gate_embedding: [sum #gates, gate_representation_dim]
    #         new_batched_reg_embedding = self.reg_conv_layers[layer_idx](graph=batched_device_dgl,
    #                                                                     feat=batched_concat_reg_embedding)
    #         new_batched_gate_embedding = self.gate_conv_layers[layer_idx](g=batched_circuit_dgl,
    #                                                                       input_gate_embedding=batched_gate_embeddings,
    #                                                                       edge_reg_embedding=edge_reg_embedding_batch)
    #         batched_reg_embeddings = new_batched_reg_embedding
    #         batched_gate_embeddings = new_batched_gate_embedding
    #
    #     # concatenate final reg representation and qubit representation
    #     # return value shape: [sum #regs, self.hidden_dimension]
    #     # reg_count_list: a list that stores #registers for each batch element
    #     assert final_reg_embedding is not None
    #     return final_reg_embedding, reg_count_list
