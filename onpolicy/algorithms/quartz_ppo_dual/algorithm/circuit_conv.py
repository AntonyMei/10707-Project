import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def circuit_message_func(edges):
    """incorporate edges' features by cat"""
    return {"neighbor_internal_embedding": torch.cat([edges.src["input_gate_embedding"],
                                                      edges.data["edge_reg_embedding"]], dim=1)}


class CircuitGraphConv(nn.Module):
    def __init__(self,
                 gate_embedding_dim,            # dimension of gate embedding
                 reg_embedding_dim,             # dimension of register embedding
                 new_gate_embedding_dim,        # dimension of new gate embedding (output)
                 ):
        super(CircuitGraphConv, self).__init__()
        # infer internal dimension from inputs
        inter_dim = 8 * math.ceil((gate_embedding_dim + reg_embedding_dim) / 16)

        # linear1: neighbor gate embedding + edge's reg embedding -> neighbor internal embedding
        self.linear1 = nn.Linear(gate_embedding_dim + reg_embedding_dim, inter_dim, bias=False)
        # linear2: neighbor internal embedding + cur gate embedding -> new gate embedding
        self.linear2 = nn.Linear(gate_embedding_dim + inter_dim, new_gate_embedding_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.01)
        nn.init.xavier_normal_(self.linear1.weight, gain=gain)
        nn.init.xavier_normal_(self.linear2.weight, gain=gain)
        nn.init.zeros_(self.linear2.bias)

    def reduce_func(self, nodes):
        # msg_dim = gate_embedding_dim + reg_embedding_dim
        # nodes.mailbox["neighbor_internal_embedding"]: [#nodes, #neighbors, msg_dim]
        # mean_neighbor_internal_embedding: [#nodes, inter_dim]
        tmp = self.linear1(nodes.mailbox["neighbor_internal_embedding"])
        tmp = F.leaky_relu(tmp, negative_slope=0.01)
        mean_neighbor_internal_embedding = torch.mean(tmp, dim=1)
        return {"mean_neighbor_internal_embedding": mean_neighbor_internal_embedding}

    def forward(self,
                g: dgl.DGLGraph,                    # the graph to operate on
                input_gate_embedding,               # input gate embedding: [#nodes, gate_embedding_dim]
                edge_reg_embedding,                 # reg embedding for each edge: [#edges, reg_embedding_dim]
                ):
        # set embeddings for current graph
        g.ndata["input_gate_embedding"] = input_gate_embedding
        g.edata["edge_reg_embedding"] = edge_reg_embedding

        # gather internal embedding of neighbors for each node
        # mean_neighbor_internal_embedding: [#nodes, inter_dim]
        g.update_all(circuit_message_func, self.reduce_func)
        mean_neighbor_internal_embedding = g.ndata["mean_neighbor_internal_embedding"]

        # compute new gate embedding
        # concat_gate_embedding: [#nodes, gate_embedding_dim + inter_dim]
        # new_gate_embedding: [#nodes, new_gate_embedding_dim]
        concat_gate_embedding = torch.cat([input_gate_embedding, mean_neighbor_internal_embedding], dim=1)
        h_linear = self.linear2(concat_gate_embedding)
        new_gate_embedding = F.leaky_relu(h_linear, negative_slope=0.01)

        # new_gate_embedding: [#nodes, new_gate_embedding_dim]
        return new_gate_embedding
