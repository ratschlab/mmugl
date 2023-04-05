# ===============================================
#
# Torch Modules for various functions
#
# ===============================================
from itertools import repeat
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP module with ReLU activation"""

    def __init__(self, dims: List[int], dropout: float = 0.0):
        """Constructor for `MLP` module"""
        super(MLP, self).__init__()

        assert len(dims) >= 2, "Class MLP requires at least 2 entries for `dims`"

        layers: List[Any] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CompressionMemoryBank(nn.Module):
    """
    Attention based compressing memory module
    """

    def __init__(self, memory_size: int, embedding_dim: int):
        """
        Constructor for `CompressionMemoryBank`

        Parameter
        ---------
        memory_size: number of memory vectors
        embedding_dim: size of the memory vectors
        """

        super(CompressionMemoryBank, self).__init__()

        self.memory_bank = nn.Parameter(torch.zeros(memory_size, embedding_dim), requires_grad=True)

        self.init_params()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # compute alignment scores `attention`
        # of memory to input
        alignment_scores = self.memory_bank @ x.transpose(0, 1)
        alignment_probabilites = torch.softmax(alignment_scores, dim=1)

        # for each memory vector extract a
        # convex combination of the input
        retrieved_memory = alignment_probabilites @ x

        return retrieved_memory

    def get_memory_bank(self) -> torch.Tensor:
        """Retrieve the internal memory vectors"""
        return self.memory_bank

    def init_params(self):
        """Initialize parameters of the internal memory bank"""
        nn.init.normal_(self.memory_bank)


class AsymmetricMemoryAttention(nn.Module):
    """
    Perform asymmetric attention onto a memory compression module
    """

    def __init__(self, memory_size: int, embedding_dim: int, heads: int = 1, dropout: float = 0.0):
        """
        Constructor for `AsymmetricMemoryAttention`

        Parameter
        ---------
        memory_size: number of memory vectors
        embedding_dim: size of the memory vectors
        heads: num. attention heads
        dropout: attention dropout
        """

        super(AsymmetricMemoryAttention, self).__init__()

        self.memory = CompressionMemoryBank(memory_size, embedding_dim)
        self.asym_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            kdim=embedding_dim,
            vdim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=False,
        )

    def forward(self, x: torch.Tensor, x_memory: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Parameter
        ---------
        x: (B, E)
            input tensor that gets projected onto
            the latent space of the compressed memory
        x_memory: (N, E)
            data to compress into the memory
        """

        # retrieve memory
        x_memory = self.memory(x_memory)
        # x_memory: (N_memory, E)

        # asymmetric attention onto compressed memory
        x_latent, _ = self.asym_attention(
            x.unsqueeze(1),  # query
            x_memory.unsqueeze(1),
            x_memory.unsqueeze(1),  # key/value
            need_weights=False,
        )
        # x_latent: (B, 1, E)

        return x_latent.squeeze(1)


class GatedGraphLookup(nn.Module):
    """
    Perform a graph `lookup` (asym. attention) onto
    prescription and disease embeddings and use
    a gated residual connection to add information
    """

    def __init__(
        self,
        graph,
        graph_nodes: List[str],
        embedding_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        """
        Constructor for `AsymmetricMemoryAttention`

        Parameter
        ---------
        embedding_dim: size of the memory vectors
        heads: num. attention heads
        dropout: attention dropout
        """
        super(GatedGraphLookup, self).__init__()

        self.graph = graph
        self.graph_nodes = graph_nodes
        self.embedding_dim = embedding_dim

        self.node_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in graph_nodes]
        )

        self.asym_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            kdim=embedding_dim,
            vdim=embedding_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=False,
        )

        self.gate_logit = nn.Linear(2 * embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass

        Parameter
        ---------
        x: (B, E)
            input tensor that is for for the lookup
            and enhanced using the retrieved embeddings
        """
        batch_dim = x.shape[0]

        graph_embeddings = self.graph.get_all_graph_embeddings()
        graph_embeddings = [
            projection(graph_embeddings[node_type])
            for projection, node_type in zip(self.node_projections, self.graph_nodes)
        ]
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        # graph_embeddings: (Nodes, Graph_E)

        # asymmetric attention onto graph embeddings
        x_lookup, _ = self.asym_attention(
            x.unsqueeze(0),  # query, single sequence element (batch_first=False)
            graph_embeddings.unsqueeze(1).expand(-1, batch_dim, -1),  # key
            graph_embeddings.unsqueeze(1).expand(-1, batch_dim, -1),  # value
            need_weights=False,
        )
        # x_lookup: (B, 1, E)

        x_lookup = x_lookup.squeeze(0)
        # x_lookup: (B, E)

        # compute gate activation
        gate_activation = torch.sigmoid(self.gate_logit(torch.cat([x, x_lookup], dim=1))).expand(
            -1, self.embedding_dim
        )
        # gate_activation: (B, E)

        # apply gate
        x_lookup = x_lookup * gate_activation
        # x_lookup: (B, E)

        # residual add gate result
        x = x + x_lookup
        # x: (B, E)

        return x
