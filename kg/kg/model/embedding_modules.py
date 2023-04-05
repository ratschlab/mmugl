# ===============================================
#
# Torch Embedding Learning Modules
#
# Some modules have been adapted from: https://github.com/jshang123/G-Bert
# ===============================================
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv

from kg.data.graph import Vocabulary, build_atc_tree, build_icd9_tree
from kg.model.graph_modules import OntologyEmbedding


class VocabularyEmbedding(nn.Module):
    """
    Embedding layer for the full vocabulary (diagnosis vocabulary and special tokens)

    Attributes
    ----------
    special_embedding: embeddings for special tokens
    disease_embedding: `OntologyEmbedding` for disease codes (GNN)
    """

    def __init__(
        self,
        disease_vocabulary: Vocabulary,
        prescription_vocabulary: Optional[Vocabulary] = None,
        num_special_tokens: int = 3,
        hidden_dims: List[int] = [128],
        disease_hidden_dim: int = 128,
        graph_staged: bool = False,
        convolution_operator: str = "GCNConv",
        attention_heads: int = 1,
    ):
        """
        Constructor for `VocabularyEmbedding`

        Parameters
        ----------
        disease_vocabulary: vocabulary of all disease codes only
        num_special_tokens: number of special tokens for the add. embeddings
        hidden_dim: hidden size within/after GNN layers and special embeddings
        disease_hidden_dim: hidden size of the disease embeddings
        graph_staged: whether the GNN is staged (G-Bert) or undirected
        convolution_operator: GNN operator to use
        attention_heads: att. heads if using e.g. GATConv
        """
        super(VocabularyEmbedding, self).__init__()

        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(torch.Tensor(num_special_tokens, hidden_dims[-1]))

        self.disease_embedding = OntologyEmbedding(
            disease_vocabulary,
            build_icd9_tree,
            in_channels=disease_hidden_dim,
            out_channels=hidden_dims,
            staged=graph_staged,
            convolution_operator=convolution_operator,
            attention_heads=attention_heads,
        )

        self.prescription_embedding = None
        if prescription_vocabulary is not None:
            self.prescription_embedding = OntologyEmbedding(
                prescription_vocabulary,
                build_atc_tree,
                in_channels=disease_hidden_dim,
                out_channels=hidden_dims,
                staged=graph_staged,
                convolution_operator=convolution_operator,
                attention_heads=attention_heads,
            )

        self.init_params()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup"""
        if self.prescription_embedding is None:
            emb = torch.cat([self.special_embedding, self.disease_embedding()], dim=0)
        else:
            emb = torch.cat(
                [
                    self.special_embedding,
                    self.disease_embedding(),
                    self.prescription_embedding(),
                ],
                dim=0,
            )

        return emb[input_ids]

    def init_params(self):
        """Initialize custom parameters"""
        torch_geometric.nn.inits.glorot(self.special_embedding)


class FlatEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_dim: int,
    ):
        super(FlatEmbedding, self).__init__()
        self.embedding = nn.Embedding(embedding_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Embedding lookup

        TODO: implement caching
        """
        return self.embedding(input_ids)
