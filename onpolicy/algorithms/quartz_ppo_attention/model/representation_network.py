from quartz import PyGraphState, PyMappingTable
import torch
import torch.nn as nn
import numpy as np
import dgl
from dgl import DGLGraph
from onpolicy.algorithms.quartz_ppo.model.circuit_gnn import CircuitGNN
from onpolicy.algorithms.quartz_ppo.model.device_gnn import DeviceGNNGINLocal, DeviceEmbedding
from onpolicy.algorithms.quartz_ppo.model.multihead_self_attention import EncoderLayer
from onpolicy.algorithms.quartz_ppo.utils.utils import graph_state_2_dgl


class RepresentationNetwork(nn.Module):
    def __init__(self,
                 # DeviceGNN
                 device_feature_type,               # degree / id / both
                 num_degree_types,                  # max degree + 1
                 num_id_types,                      # max node id
                 degree_embedding_dim,              # dimension of feature embedding
                 id_embedding_dim,                  # dimension of id embedding
                 device_num_layers,                 # number of gnn convolution layers
                 device_hidden_dimension,           # dimension of each internal GNN layer
                 device_out_dimension,              # output dimension of final GNN layer
                 # Circuit GNN
                 circuit_num_layers,                # number of gnn convolution layers
                 num_gate_types,                    # number of different gate types
                 gate_type_embedding_dim,           # dimension of gate type embedding
                 circuit_conv_internal_dim,         # hidden layer dimension of each convolution layer's MLP
                 circuit_out_dimension,             # output dimension of final GNN layer
                 # Multi-head self-attention
                 final_mlp_hidden_dimension_ratio,  # final MLP's hidden dimension / raw representation dimension
                 num_attention_heads,               # number of attention heads
                 attention_qk_dimension,            # dimension of q vector and k vector in attention
                 attention_v_dimension,             # dimension of v vector in attention
                 # device
                 device,                            # device the network is on
                 ):
        super(RepresentationNetwork, self).__init__()

        # device gnn and circuit gnn
        self.device = device
        self.device_gnn = DeviceGNNGINLocal(feature_type=device_feature_type, num_degree_types=num_degree_types,
                                            num_id_types=num_id_types, degree_embedding_dim=degree_embedding_dim,
                                            id_embedding_dim=id_embedding_dim, num_layers=device_num_layers,
                                            hidden_dimension=device_hidden_dimension,
                                            out_dimension=device_out_dimension)
        self.circuit_gnn = CircuitGNN(num_layers=circuit_num_layers, num_gate_types=num_gate_types,
                                      gate_type_embed_dim=gate_type_embedding_dim, h_feats=circuit_out_dimension,
                                      inter_dim=circuit_conv_internal_dim)

        # multi-head self attention
        raw_rep_dimension = device_out_dimension + circuit_out_dimension
        self.attention_encoder = EncoderLayer(d_model=raw_rep_dimension,
                                              d_inner=raw_rep_dimension * final_mlp_hidden_dimension_ratio,
                                              n_head=num_attention_heads,
                                              d_k=attention_qk_dimension, d_v=attention_v_dimension)

    def forward(self):
        pass


class RepresentationNetworkSimple(nn.Module):
    def __init__(self,
                 # DeviceGNN
                 num_registers,                     # number of registers
                 device_out_dimension,              # output dimension of device embedding network
                 # Circuit GNN
                 circuit_num_layers,                # number of gnn convolution layers
                 num_gate_types,                    # number of different gate types
                 gate_type_embedding_dim,           # dimension of gate type embedding
                 circuit_conv_internal_dim,         # hidden layer dimension of each convolution layer's MLP
                 circuit_out_dimension,             # output dimension of final GNN layer
                 # Multi-head self-attention
                 final_mlp_hidden_dimension_ratio,  # final MLP's hidden dimension / raw representation dimension
                 num_attention_heads,               # number of attention heads
                 attention_qk_dimension,            # dimension of q vector and k vector in attention
                 attention_v_dimension,             # dimension of v vector in attention
                 # device
                 device,                            # device the network is on
                 ):
        super(RepresentationNetworkSimple, self).__init__()
        self.device = device

        # device gnn and circuit gnn
        self.device_embedding_network = DeviceEmbedding(num_registers=num_registers,
                                                        embedding_dimension=device_out_dimension)
        self.circuit_gnn = CircuitGNN(num_layers=circuit_num_layers, num_gate_types=num_gate_types,
                                      gate_type_embed_dim=gate_type_embedding_dim, h_feats=circuit_out_dimension,
                                      inter_dim=circuit_conv_internal_dim)

        # multi-head self attention
        self.device_out_dimension = device_out_dimension
        self.circuit_out_dimension = circuit_out_dimension
        self.raw_rep_dimension = device_out_dimension + circuit_out_dimension
        self.attention_encoder = EncoderLayer(d_model=self.raw_rep_dimension,
                                              d_inner=self.raw_rep_dimension * final_mlp_hidden_dimension_ratio,
                                              n_head=num_attention_heads,
                                              d_k=attention_qk_dimension, d_v=attention_v_dimension)

    def forward(self, circuit_batch: [PyGraphState], physical2logical_mapping_batch: [PyMappingTable]):
        assert len(circuit_batch) == len(physical2logical_mapping_batch)

        # recover circuit dgl and do batch inference for circuit representation
        circuit_dgl_list, gate_count_list, qubit_count_list = [], [], []
        for circuit in circuit_batch:
            # circuit dgl
            circuit_dgl = graph_state_2_dgl(circuit, device=self.device)
            circuit_dgl_list.append(circuit_dgl)
            # gate count
            gate_count = len(circuit.is_input)
            gate_count_list.append(gate_count)
            # qubit count
            num_qubits = sum(circuit.is_input)
            qubit_count_list.append(num_qubits)
        batched_circuit_dgl = dgl.batch(circuit_dgl_list)
        batched_rep = self.circuit_gnn(batched_circuit_dgl)

        # cut batched rep to recover list structure
        logical_qubit_rep_list = []
        cur_pos = 0
        for gate_count, qubit_count in zip(gate_count_list, qubit_count_list):
            cur_logical_qubit_rep = batched_rep[cur_pos: cur_pos + qubit_count]
            logical_qubit_rep_list.append(cur_logical_qubit_rep)
            cur_pos += gate_count
        assert cur_pos == batched_rep.shape[0]

        # generate batched circuit representation using circuit GNN
        concatenated_raw_rep_list = []
        for circuit, logical_qubit_rep, num_qubits, physical2logical_mapping in \
                zip(circuit_batch, logical_qubit_rep_list, qubit_count_list, physical2logical_mapping_batch):

            # append empty representation
            num_registers = len(physical2logical_mapping)
            logical_qubit_rep_padding = torch.zeros(num_registers - num_qubits, self.circuit_out_dimension,
                                                    device=self.device)
            logical_qubit_rep = torch.concat([logical_qubit_rep, logical_qubit_rep_padding], dim=0)

            # reorder circuit representation
            # (This inversion is necessary because logical id may not be continuous)
            qubit_physical_idx = list(circuit.input_physical_idx[:num_qubits])
            for i in range(num_registers):
                if i not in qubit_physical_idx:
                    qubit_physical_idx.append(i)

            def invert_permutation(permutation):
                inv = np.empty_like(permutation)
                inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
                return inv

            inverted_physical_idx = np.array(invert_permutation(qubit_physical_idx))
            gather_indices = [inverted_physical_idx for _ in range(self.circuit_out_dimension)]
            gather_indices = np.array(gather_indices)
            gather_indices = torch.tensor(gather_indices, device=self.device).transpose(0, 1)
            sorted_logical_qubit_rep = torch.gather(input=logical_qubit_rep, dim=0, index=gather_indices)

            # concatenate with device embedding
            device_embedding_input = torch.tensor(list(range(num_registers)), device=self.device)
            device_embedding = self.device_embedding_network(device_embedding_input)
            concatenated_raw_rep = torch.concat([sorted_logical_qubit_rep, device_embedding], dim=1)
            concatenated_raw_rep_list.append(concatenated_raw_rep[None, :])
        concatenated_raw_rep = torch.concat(concatenated_raw_rep_list, dim=0)

        # send into self attention layer and return
        register_representation, attention_score_mat = self.attention_encoder(concatenated_raw_rep)
        attention_score_mat = torch.sum(attention_score_mat, dim=1)
        return register_representation, attention_score_mat
