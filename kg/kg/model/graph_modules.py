# ===============================================
#
# Torch Graph Learning Modules
#
# Some modules have been adapted from: https://github.com/jshang123/G-Bert
# ===============================================
import logging
import pickle
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    EdgePooling,
    GATConv,
    GCNConv,
    GINConv,
    GraphConv,
    HeteroConv,
    SAGEConv,
    to_hetero,
)
from torch_geometric.utils import dropout_adj

from kg.data.contants import ALLOWED_CO_OCCURRENCE
from kg.data.datasets import CodeTokenizer
from kg.data.graph import (
    CoLinkConfig,
    Vocabulary,
    build_atc_tree,
    build_co_occurrence_edges,
    build_co_occurrence_links,
    build_co_occurrence_vocabulary,
    build_cominbed_edges,
    build_icd9_tree,
    build_stage_one_edges,
    build_stage_two_edges,
    cluster_co_occurrence_nodes,
    load_co_occurrence_features,
)
from kg.model.contractions import edge_pooling_expand_nodes, edge_pooling_rewrite_edges
from kg.model.utility_modules import MLP
from kg.training.dataloader import InfiniteDataLoader
from kg.utils.metrics import metric_report


class OntologyEmbedding(nn.Module):
    """
    GNN over an ontology tree
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        build_tree_func: Callable,
        in_channels: int = 128,
        out_channels: List[int] = [128],
        staged: bool = False,
        convolution_operator: str = "GCNConv",
        attention_heads: int = 1,
    ):
        """
        Constructor for `OntologyEmbedding`

        Parameters
        ----------
        vocabulary: Target vocabulary of the ontology
        build_tree_func: function to build the graph from the `vocabulary`
        in_channels: -
        out_channels: -
        staged: whether to run convolutions staged (G-Bert paper)
        convolution_operator: -
        attention_heads: number of attention heads if using e.g. GATConv
        """
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        tree_mapping, graph_vocabulary = build_tree_func(list(vocabulary.idx2word.values()))
        self.graph_vocabulary = graph_vocabulary
        self.tree_mapping = tree_mapping
        self.staged = staged

        self.edges: Any = None
        if staged:
            self.edges = (
                torch.tensor(build_stage_one_edges(tree_mapping, graph_vocabulary)),
                torch.tensor(build_stage_two_edges(tree_mapping, graph_vocabulary)),
            )

        else:
            self.edges = torch.tensor(build_cominbed_edges(tree_mapping, graph_vocabulary))

        # build graph conv layer
        # G-Bert uses GATConv
        assert (
            len(out_channels) >= 1
        ), "`Ontology Embedding requires at least one out channel dimension"
        logging.info(f"`OntologyEmbedding` using {convolution_operator}")
        if convolution_operator == "GCNConv":
            self.graph_convs = nn.ModuleList(
                [
                    GCNConv(
                        in_channels=in_channels,
                        out_channels=out_channels[0],
                        cached=False,
                    )
                ]
            )
            for i in range(len(out_channels) - 1):
                self.graph_convs.append(
                    GCNConv(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i + 1],
                        cached=False,
                    )
                )
        elif convolution_operator == "GINConv":
            self.graph_convs = nn.ModuleList(
                [GINConv(MLP([in_channels, out_channels[0]]), train_eps=True)]
            )
            for i in range(len(out_channels) - 1):
                self.graph_convs.append(
                    GINConv(MLP([out_channels[i], out_channels[i + 1]]), train_eps=True)
                )
        else:

            assert (
                convolution_operator == "GATConv"
            ), f"Non-supported GNN operator {convolution_operator}"
            assert_msg = f"Graph out_channels: {out_channels} must be divisible by num heads: {attention_heads} for GATConv"
            assert all([c % attention_heads == 0 for c in out_channels]), assert_msg

            self.graph_convs = nn.ModuleList(
                [
                    GATConv(
                        in_channels=in_channels,
                        out_channels=out_channels[0] // attention_heads,
                        heads=attention_heads,
                    )
                ]
            )
            for i in range(len(out_channels) - 1):
                self.graph_convs.append(
                    GATConv(
                        in_channels=out_channels[i],
                        out_channels=out_channels[i + 1] // attention_heads,
                        heads=attention_heads,
                    )
                )

        # get number of layers
        self.graph_num_layers = len(self.graph_convs)

        # graph embedding
        self.num_nodes = len(graph_vocabulary.word2idx)
        logging.info(f"OntologyEmbedding has {self.num_nodes} nodes")
        self.embedding = nn.Parameter(torch.Tensor(self.num_nodes, in_channels))

        # idx mapping: from leaf node in `graph_vocabulary` to outer vocabulary
        self.idx_mapping = [
            self.graph_vocabulary.word2idx[word] for word in vocabulary.idx2word.values()
        ]

        self.init_params()

    def map_embeddings(self):

        emb = self.embedding
        for i, gnn_layer in enumerate(self.graph_convs):

            # run GNN layer
            emb = self.run_graph_layer(gnn_layer, emb)

            # apply ReLU after all but last layer
            if i < (self.graph_num_layers - 1):
                emb = nn.functional.relu(emb)

        return emb

    def get_all_graph_embeddings(self) -> Union[torch.Tensor, torch.nn.Parameter]:
        """
        Returns all graph embeddings after convolutions

        Returns
        -------
        Transformed graph embeddings
        """
        return self.map_embeddings()

    def run_graph_layer(self, gnn_layer: nn.Module, x: Union[torch.Tensor, torch.nn.Parameter]):

        if self.staged:
            x = gnn_layer(x, self.edges[0].to(x.device))
            x = gnn_layer(x, self.edges[1].to(x.device))
        else:
            x = gnn_layer(x, self.edges.to(x.device))

        return x

    def forward(self) -> torch.Tensor:
        """
        Computes graph convolutions and maps the embeddings
        to correspond to the vocabulary mapping of the parent layer

        Returns
        -------
        Convolved and shuffled embeddings
        """
        emb = self.map_embeddings()
        return emb[self.idx_mapping]

    def init_params(self):
        """Initializes custom layer parameters"""
        torch_geometric.nn.inits.glorot(self.embedding)

    @torch.no_grad()
    def store_embeddings(self, file_path: str):
        """
        Store embeddings and the associated Vocabulary
        and mapping

        Parameters
        ----------
        file_path: file to store the data to
        """
        embedding_data = {}
        embedding_data["embeddings"] = self.get_all_graph_embeddings().data.detach().cpu()
        embedding_data["vocabulary"] = self.graph_vocabulary
        embedding_data["mapping"] = self.tree_mapping

        with open(file_path, "wb") as handle:
            pickle.dump(embedding_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class OntologyHeteroGNN(nn.Module):
    """A GNN to be used on an `OntologyEmbedding`"""

    def __init__(
        self,
        edge_types: Sequence[Tuple[str, str, str]],
        embedding_dim: int,
        graph_num_layers: int = 1,
        convolution_operator: str = "GCNConv",
        attention_heads: int = 1,
        divisor: int = 4,
        graph_num_filters: int = 1,
        edge_weights: bool = False,
        gbert_mode: bool = False,
        contractions: List[str] = [],
        contractions_type: str = None,
    ):
        """
        Constructor for `OntologyHeteroGNN`

        Parameters
        ----------
        edge_types: A list of tuples describing
            the edge types (to be used in construction with
            a Pytorch Geometric `HeteroConv` module)
        embedding_dim: -
        graph_num_layers: -
        convolution_operator: GNN operator to use
        attention_heads: number of attention heads to use
            for some `convolution_operator`, ignored for others.
        divisor: for co-occurrence nodes with lower dimensionality
        graph_num_filters: int
            number of distinct stacks of graph layers
        edge_weights: bool
            whether edge weights are provided for `occurs_link` edges
            works only with GraphConv
        contractions: List[str]
            node types to apply contractions to
        contractions_type: str
            scoring func for contractions
        """
        super(OntologyHeteroGNN, self).__init__()

        self.gbert_mode = gbert_mode
        if gbert_mode:
            logging.error("Warning, running in Gbert Mode")

        # set parameters
        self.graph_num_layers = graph_num_layers
        self.embedding_dim = embedding_dim
        self.convolution_operator = convolution_operator
        self.attention_heads = attention_heads
        self.divisor = divisor
        self.graph_num_filters = graph_num_filters
        self.edge_weights = edge_weights

        # build graph layers
        self.graph_convs = nn.ModuleList()
        for _ in range(graph_num_filters):

            # add set of layers for each filter
            convs = nn.ModuleList()
            for _ in range(graph_num_layers):
                conv = HeteroConv(
                    {edge: self.conv_module_builder(edge) for edge in edge_types},
                    aggr="sum",
                )
                convs.append(conv)

            self.graph_convs.append(convs)

        # init contractions
        self.contractions = contractions
        self.do_contractions = len(contractions) > 0
        self.pool_stacks = nn.ModuleList()
        self.contractions_type = contractions_type

        if self.do_contractions:

            # get scoring function
            if contractions_type == "softmax":
                scoring_func = EdgePooling.compute_edge_score_softmax
            elif contractions_type == "sigmoid":
                scoring_func = EdgePooling.compute_edge_score_sigmoid
            else:
                raise ValueError(f"Scoring func {contractions_type} needs to be sigmoid or softmax")

            # create layers
            for _ in range(graph_num_filters):
                pool_layers = nn.ModuleList()
                for _ in range(graph_num_layers):
                    pool = nn.ModuleDict(
                        {
                            key: EdgePooling(
                                self.embedding_dim,
                                scoring_func,
                                dropout=0,
                                add_to_edge_score=0.0,
                            )  # 0.5
                            for key in contractions
                        }
                    )
                    pool_layers.append(pool)
                self.pool_stacks.append(pool_layers)

            logging.info(f"[{self.__class__.__name__}`] contraction score func: {scoring_func}")
            logging.info(
                f"[{self.__class__.__name__}`] Built contraction layers: {self.pool_stacks}"
            )

        # log build
        logging.info(f"Built: {self}")

    def apply_contraction(
        self,
        pool_layers: nn.ModuleDict,
        x_dict: Dict,
        edge_index_dict: Dict,
        edge_weights_dict: Dict = None,
    ):
        """
        Apply a single contraction layer across
        all types to be contracted, adjust all edges accordingly

        Parameter
        ---------
        pool_layers:
            the layers performing the contraction
        x_dict:
            node embedding dict
        edge_index_dict:
            edge indeces dict
        edge_weights_dict:
            edge weights dict
        """

        # bookkeeping
        clustering_dict = {}

        for node_type in self.contractions:

            target_nodes = x_dict[node_type]
            target_edges = edge_index_dict[(node_type, "tree", node_type)]
            batch_index = torch.zeros(
                target_nodes.shape[0], dtype=torch.int64, device=target_nodes.device
            )

            contracted_nodes, _, _, unpool_info = pool_layers[node_type](
                target_nodes, target_edges, batch_index
            )
            clustering_dict[node_type] = unpool_info.cluster

            x_dict[node_type] = contracted_nodes

            # rewrite edges
            # `both` edges
            for edge in edge_index_dict.keys():

                # get edge weights
                if edge_weights_dict is None:
                    edge_weights = None
                else:
                    edge_weights = edge_weights_dict[edge]

                # both endpoints affected by pooling
                new_edge_index = None
                if edge[0] == edge[2] and edge[0] == node_type:

                    new_edge_index, new_edge_weights = edge_pooling_rewrite_edges(
                        edge_index_dict[edge],
                        unpool_info.cluster,
                        edge_weights=edge_weights,
                        endpoint_indeces=(0, 1),
                        rm_self_loops=True,
                        rm_duplicates=True,
                        coalesce_reduce="max",
                    )

                # only source affected by pooling
                elif edge[0] == node_type:

                    new_edge_index, new_edge_weights = edge_pooling_rewrite_edges(
                        edge_index_dict[edge],
                        unpool_info.cluster,
                        edge_weights=edge_weights,
                        endpoint_indeces=(0,),
                        rm_self_loops=False,
                        rm_duplicates=True,
                        coalesce_reduce="max",
                    )

                # only destination affected by pooling
                elif edge[2] == node_type:

                    new_edge_index, new_edge_weights = edge_pooling_rewrite_edges(
                        edge_index_dict[edge],
                        unpool_info.cluster,
                        edge_weights=edge_weights,
                        endpoint_indeces=(1,),
                        rm_self_loops=False,
                        rm_duplicates=True,
                        coalesce_reduce="max",
                    )

                # set new edge indeces and weights
                if new_edge_index is not None:
                    edge_index_dict[edge] = new_edge_index

                    if edge_weights_dict is not None:
                        edge_weights_dict[edge] = new_edge_weights

        return x_dict, edge_index_dict, edge_weights_dict, clustering_dict

    def conv_module_builder(self, edge):

        # compute channel dimensions
        # to handle lower dim co-rce nodes
        lower_dim = int(self.embedding_dim // self.divisor)
        if edge[0] in ALLOWED_CO_OCCURRENCE or edge[2] in ALLOWED_CO_OCCURRENCE:
            if edge[0] in ALLOWED_CO_OCCURRENCE:
                in_channels = (lower_dim, self.embedding_dim)
                out_channels = self.embedding_dim
            else:
                in_channels = (self.embedding_dim, lower_dim)
                out_channels = lower_dim
        else:
            in_channels = out_channels = self.embedding_dim

        if self.convolution_operator == "GCNConv":
            return GCNConv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                cached=False, 
            )

        elif self.convolution_operator == "GINConv":
            in_c = in_channels[0] if isinstance(in_channels, tuple) else in_channels

            logging.debug(f"[GINConv] Edge {edge}: {in_c} to {out_channels}")
            return GINConv(MLP([in_c, self.embedding_dim, out_channels]), train_eps=True)

        elif self.convolution_operator == "GATConv":

            in_channels = (
                int(in_channels // self.attention_heads) if self.gbert_mode else in_channels
            )
            out_channels = (
                int(out_channels // self.attention_heads) if self.gbert_mode else out_channels
            )
            logging.debug(f"[GATConv] Edge {edge}: {in_channels} to {out_channels}")

            return GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                concat=self.gbert_mode,  # average/concat attention heads
                heads=self.attention_heads,
            )

        elif self.convolution_operator == "SAGEConv":

            logging.debug(f"[SAGEConv] Edge {edge}: {in_channels} to {out_channels}")

            return SAGEConv(
                in_channels=in_channels,
                out_channels=out_channels,
                aggr="mean",  # max, mean, add, lstm
                normalize=False,
            )

        elif self.convolution_operator == "GraphConv":

            aggr = "mean"
            if self.edge_weights and edge[1] == "occurs_link":
                aggr = "add"
            logging.debug(f"[GraphConv] Edge {edge}: aggregation: {aggr}")

            return GraphConv(in_channels=in_channels, out_channels=out_channels, aggr=aggr)

        else:
            raise NotImplementedError(f"Non-supported GNN operator {self.convolution_operator}")

    def legacy_forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, ...], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        for i, conv in enumerate(self.graph_convs):

            # graph conv layer
            x_dict = conv(x_dict, edge_index_dict)

            # apply ReLU after all but last layer
            if i < (self.graph_num_layers - 1):
                x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, ...], torch.Tensor],
        edge_weights: Optional[Dict[Tuple[str, ...], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        if not hasattr(self, "graph_num_filters"):
            return self.legacy_forward(x_dict, edge_index_dict)

        # get input for each stack of layers
        activations = [x_dict for _ in range(self.graph_num_filters)]

        # go over each stack of layers
        for l, layers in enumerate(self.graph_convs):

            # book-keeping of contractions
            clustering_layers = []

            # single stack of layers
            for i, conv in enumerate(layers):

                # graph conv layer
                if edge_weights is None:
                    activations[l] = conv(activations[l], edge_index_dict)
                else:
                    activations[l] = conv(activations[l], edge_index_dict, edge_weights)

                # apply ReLU after all but last layer
                if i < (self.graph_num_layers - 1):
                    activations[l] = {key: x.relu() for key, x in activations[l].items()}

                # pooling / contractions
                if hasattr(self, "do_contractions") and self.do_contractions:
                    (
                        activations[l],
                        edge_index_dict,
                        edge_weights,
                        clustering_dict,
                    ) = self.apply_contraction(
                        self.pool_stacks[l][i],
                        activations[l],
                        edge_index_dict,
                        edge_weights,
                    )
                    clustering_layers.append(clustering_dict)

            # retrace contractions and expand node embeddings
            if hasattr(self, "do_contractions") and self.do_contractions:
                for node_type in self.contractions:
                    for clustering in reversed(clustering_layers):

                        # before_shape = activations[l][node_type].shape
                        activations[l][node_type] = edge_pooling_expand_nodes(
                            activations[l][node_type], clustering[node_type]
                        )

        # pool the features
        x_dict = {
            key: torch.max(
                torch.stack([activations[l][key] for l in range(self.graph_num_filters)]),
                dim=0,
            ).values
            for key in x_dict.keys()
        }

        return x_dict

    def __str__(self) -> str:
        heads = "" if self.attention_heads == 1 else f", heads: {self.attention_heads}"
        filters = 0 if not hasattr(self, "graph_num_filters") else self.graph_num_filters
        return f"`{self.__class__.__name__}` {self.graph_num_layers} layers, {filters} filters: {self.convolution_operator}:{self.embedding_dim}{heads}"


class HeterogenousOntologyEmbedding(nn.Module):
    """
    Heterogenous Ontology Embedding over multiple
    ontologies and interconnecting patient
    co-occurrences
    """

    def __init__(
        self,
        disease_vocabulary: Vocabulary,
        prescription_vocabulary: Vocabulary,
        embedding_dim: int,
        graph_num_layers: int,
        graph_num_filters: int = 1,
        convolution_operator: str = "GCNConv",
        attention_heads: int = 1,
        num_special_tokens: int = 3,
        data_pd: Optional[pd.DataFrame] = None,
        co_occurrence: Optional[str] = None,
        co_occurrence_subsample: float = 0.3,
        co_occurrence_loss: float = 0.0,
        co_occurrence_dropout: float = 0.0,
        co_occurrence_cluster: int = 0,
        co_occurrence_features: Optional[str] = None,
        co_occurrence_batch_size: int = 16,
        tokenizer: Optional[CodeTokenizer] = None,
        divisor: int = 4,
        triplet_loss: float = 0.0,
        triplet_batch_size: int = 16,
        triplet_margin: float = 0.1,
        gbert_mode: bool = False,
        co_link_config: CoLinkConfig = None,
        contractions_type: str = None,
        trainable_edge_weights: bool = False,
    ):
        """
        Constructor for `HeterogenousOntologyEmbedding`

        Parameter
        ---------
        disease_vocabulary: ICD code vocabulary to use
        prescription_vocabulary: ATC code vocabulary to use
        tokenizer_vocabulary: full vocabulary used for tokenization
        embedding_dim: dimension of the embeddings
        graph_num_layers: number of graph layers to stack
        graph_num_filters: int
            number of distinct stacks of graph layers
        convolution_operator: GNN operator selector
        attention_heads: number of attention heads for some GNN operators
        data_pd: patient records from MIMIC for co-occurrence graph
        co_occurrence: co-occurrence node type, one of {visit, patient}
        co_occurrence_subsample: subsampling ratio [0, 1] for co-occurrence nodes
        co_occurrence_loss: alpha parameter for the additional co-occurrence
            node autoencoder loss
        co_occurrence_dropout: dropout to apply on the edges towards the co-nodes
        co_occurrence_cluster: reduce num co nodes by clustering
        co_occurrence_features: path to DataFrame with stored static node features
        co_occurrence_batch_size: training batch size for the co-occurrence loss
        tokenizer: `CodeTokenizer` for model
        divisor: for co-occurrence nodes with lower dimensionality
        triplet_loss: alpha parameter for the additional triplet loss
        triplet_batch_size: -
        triplet_margin: margin parameter of the triplet loss
        contractions_type: str
            perform graph contractions i.e. edge pooling with prov. score func
        trainable_edge_weights: bool
            make edge weights trainable
        """
        super(HeterogenousOntologyEmbedding, self).__init__()

        if co_occurrence_loss > 0.0:
            assert tokenizer is not None, "Need to pass tokenizer if co-occurrence used"
            self.tokenizer = tokenizer

        self.co_occurrence_dropout = co_occurrence_dropout
        self.co_occurrence = co_occurrence
        self.co_occurrence_batch_size = co_occurrence_batch_size

        if gbert_mode:
            logging.error("Warning, running in Gbert Mode")

        if co_link_config is not None:
            assert convolution_operator in {
                "GraphConv",
                "GCNConv",
            }, f"Cannot use {convolution_operator} with co links"

        self.trainable_edge_weights = trainable_edge_weights
        logging.info(f"[GRAPH] edge weights trainable: {self.trainable_edge_weights}")
        if trainable_edge_weights:
            assert_msg = "Trainable edge weights requires: GraphConv and co occurrence edges"
            assert convolution_operator == "GraphConv" and co_link_config is not None, assert_msg

        # Build the two hierarchical ontology trees
        disease_tree, disease_graph_voc = build_icd9_tree(
            list(disease_vocabulary.idx2word.values())
        )
        self.disease_graph_voc = disease_graph_voc

        prescription_tree, prescription_graph_voc = build_atc_tree(
            list(prescription_vocabulary.idx2word.values())
        )
        self.prescription_graph_voc = prescription_graph_voc

        d_edges = torch.tensor(build_cominbed_edges(disease_tree, disease_graph_voc))
        self.disease_edges = nn.Parameter(d_edges, requires_grad=False)

        p_edges = torch.tensor(build_cominbed_edges(prescription_tree, prescription_graph_voc))
        self.prescription_edges = nn.Parameter(p_edges, requires_grad=False)

        # initialize the node representations
        code_embedding_dim = int(embedding_dim // attention_heads) if gbert_mode else embedding_dim
        num_disease_nodes = len(disease_graph_voc.word2idx)
        self.disease_embedding = nn.Parameter(torch.Tensor(num_disease_nodes, code_embedding_dim))

        num_prescription_nodes = len(prescription_graph_voc.word2idx)
        self.prescription_embedding = nn.Parameter(
            torch.Tensor(num_prescription_nodes, code_embedding_dim)
        )
        logging.info(
            f"{self.__class__.__name__} has {num_disease_nodes} disease nodes and {num_prescription_nodes} prescription nodes"
        )

        # idx mapping: from leaf node in `graph_vocabulary` to outer vocabulary
        self.disease_idx_mapping = [
            disease_graph_voc.word2idx[word] for word in disease_vocabulary.idx2word.values()
        ]
        self.prescription_idx_mapping = [
            prescription_graph_voc.word2idx[word]
            for word in prescription_vocabulary.idx2word.values()
        ]

        # initialize the embeddings for special tokens
        self.special_embedding = nn.Parameter(torch.Tensor(num_special_tokens, embedding_dim))

        # Co-occurrence
        self.co_occurrence_voc = None
        if data_pd is not None and co_occurrence is not None:
            logging.info("[GRAPH] building co-occurrence graph")
            assert convolution_operator not in [
                "GCNConv"
            ], f"{convolution_operator} not supported for bipartite graph"
            self.co_occurrence_graph = True

            # build node vocabulary
            self.co_occurrence_voc = build_co_occurrence_vocabulary(
                data_pd, co_occurrence, subsample=co_occurrence_subsample
            )

            # reduce vocabulary size by clustering
            if co_occurrence_cluster > 0 and tokenizer is not None:
                self.co_occurrence_voc = cluster_co_occurrence_nodes(
                    data_pd,
                    self.co_occurrence_voc,
                    tokenizer,
                    co_occurrence,
                    co_occurrence_cluster,
                )

            # build edges
            edges = build_co_occurrence_edges(
                data_pd,
                self.co_occurrence_voc,  # type: ignore
                disease_graph_voc,
                prescription_graph_voc,
                co_occurrence,
                co_occurrence_loss=co_occurrence_loss,
                tokenizer=tokenizer,
            )

            self.d2node = nn.Parameter(torch.tensor(edges["disease2node"]), requires_grad=False)
            self.node2d = nn.Parameter(torch.tensor(edges["node2disease"]), requires_grad=False)
            self.p2node = nn.Parameter(
                torch.tensor(edges["prescription2node"]), requires_grad=False
            )
            self.node2p = nn.Parameter(
                torch.tensor(edges["node2prescription"]), requires_grad=False
            )

            # build co-occurrence node embeddings
            num_nodes = len(self.co_occurrence_voc.word2idx)  # type: ignore
            if co_occurrence_features is not None:  # build from features
                self.co_occurrence_static = True

                # get static node features
                assert self.co_occurrence_voc is not None  # mypy
                self.co_occurrence_nodes = load_co_occurrence_features(
                    co_occurrence_features,
                    self.co_occurrence_voc,
                    data_pd,
                    co_occurrence,
                )  # type: ignore

                # additional projection layer to go from static feature
                # space into graph embedding dimensional space
                self.node_feature_projector = nn.Linear(
                    self.co_occurrence_nodes.shape[1], int(embedding_dim // divisor)
                )

                logging.info(
                    f"[GRAPH] co-occurrence nodes (features, static): {self.co_occurrence_nodes.shape}"
                )

            else:  # trainable embeddings
                self.co_occurrence_static = False
                self.co_occurrence_nodes = nn.Parameter(
                    torch.Tensor(num_nodes, int(embedding_dim // divisor))
                )
                logging.info(
                    f"[GRAPH] co-occurrence nodes (random init, trainable): {self.co_occurrence_nodes.shape}"
                )

            # setup loss targets for co occurrence node auto-encoder
            if co_occurrence_loss > 0.0:
                self.co_occurrence = co_occurrence
                self.co_occurrence_loss = co_occurrence_loss
                d_nodes = len(tokenizer.disease_vocabulary.word2idx)  # type: ignore
                p_nodes = len(tokenizer.prescription_vocabulary.word2idx)  # type: ignore

                self.co_d_target = edges["d_targets"]
                self.co_p_target = edges["p_targets"]
                self.co_d_clf = MLP(
                    [int(embedding_dim // divisor)] + list(repeat(128, 2)) + [d_nodes]
                )
                self.co_p_clf = MLP(
                    [int(embedding_dim // divisor)] + list(repeat(128, 2)) + [p_nodes]
                )

                # get split
                TEST_SIZE = 0.3
                self.co_train_ids, self.co_val_ids = train_test_split(
                    torch.tensor(list(range(num_nodes)), dtype=torch.int64),
                    test_size=TEST_SIZE,
                    shuffle=True,
                )
                logging.info(
                    f"[GRAPH] occurence nodes, train: {len(self.co_train_ids)}, val: {len(self.co_val_ids)}"
                )

                # get split masks
                self.co_train_mask = self.co_train_ids.new_empty(
                    len(self.co_occurrence_nodes), dtype=torch.bool
                )
                self.co_train_mask.fill_(False)
                self.co_train_mask[self.co_train_ids] = True

                self.co_val_mask = self.co_val_ids.new_empty(
                    len(self.co_occurrence_nodes), dtype=torch.bool
                )
                self.co_val_mask.fill_(False)
                self.co_val_mask[self.co_val_ids] = True

                logging.info(
                    f"[GRAPH] occurence nodes, train: {len(self.co_train_ids)} ({self.co_train_mask.sum()}), val: {len(self.co_val_ids)} ({self.co_val_mask.sum()})"
                )

                # co classification AE dataloader
                logging.info(f"[TRAIN] co batch size: {self.co_occurrence_batch_size}")
                self.co_train_loader = iter(
                    InfiniteDataLoader(
                        TensorDataset(self.co_train_ids),
                        shuffle=True,
                        batch_size=co_occurrence_batch_size,
                    )
                )

        else:
            self.co_occurrence_graph = False
            self.co_occurrence_static = False

        # build co occurrence links
        self.co_occurrence_links = False
        if co_link_config is not None and data_pd is not None:
            self.co_occurrence_links = True
            logging.info(f"[GRAPH] co links: {co_link_config}")

            # get vocabulary or build new
            if self.co_occurrence_voc is not None and co_occurrence == co_link_config.link_type:
                self.co_link_voc = self.co_occurrence_voc
            else:
                self.co_link_voc = build_co_occurrence_vocabulary(
                    data_pd, co_link_config.link_type, subsample=co_occurrence_subsample
                )

            # build link edges
            edge_data = build_co_occurrence_links(
                data_pd,
                self.disease_graph_voc,
                self.prescription_graph_voc,
                self.co_link_voc,
                co_link_config.link_type,
                co_link_config.edge_weights,
                co_link_config.normalize_weights,
            )

            # create edge/weight tensors
            self.d2d_co_edges = nn.Parameter(torch.tensor(edge_data["d2d"]), requires_grad=False)
            self.p2p_co_edges = nn.Parameter(torch.tensor(edge_data["p2p"]), requires_grad=False)
            self.d2p_co_edges = nn.Parameter(torch.tensor(edge_data["d2p"]), requires_grad=False)
            self.p2d_co_edges = nn.Parameter(torch.tensor(edge_data["p2d"]), requires_grad=False)

            if co_link_config.edge_weights:

                self.d2d_co_weights = nn.Parameter(
                    edge_data["d2d_weights"] * co_link_config.alpha_intra,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore
                self.p2p_co_weights = nn.Parameter(
                    edge_data["p2p_weights"] * co_link_config.alpha_intra,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore
                self.d2p_co_weights = nn.Parameter(
                    edge_data["d2p_weights"] * co_link_config.alpha_inter,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore
                self.p2d_co_weights = nn.Parameter(
                    edge_data["p2d_weights"] * co_link_config.alpha_inter,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore

        # assemble heterogeneous graph data
        self.graph_data = HeteroData()
        self.graph_data["disease"].x = self.disease_embedding
        self.graph_data["prescription"].x = self.prescription_embedding

        self.graph_data["disease", "tree", "disease"].edge_index = self.disease_edges
        self.graph_data["prescription", "tree", "prescription"].edge_index = self.prescription_edges

        # gather edge types
        edge_types = [
            ("disease", "tree", "disease"),
            ("prescription", "tree", "prescription"),
        ]

        # add co-occurrence data components
        if self.co_occurrence_graph:

            # add nodes
            self.graph_data[co_occurrence].x = self.co_occurrence_nodes

            # add edges in all directions
            self.graph_data[co_occurrence, "occurs", "disease"].edge_index = self.node2d
            self.graph_data["disease", "occurs", co_occurrence].edge_index = self.d2node
            self.graph_data[co_occurrence, "occurs", "prescription"].edge_index = self.node2p
            self.graph_data["prescription", "occurs", co_occurrence].edge_index = self.p2node

            # add additional edge types
            edge_types.append((co_occurrence, "occurs", "disease"))  # type: ignore
            edge_types.append(("disease", "occurs", co_occurrence))  # type: ignore
            edge_types.append((co_occurrence, "occurs", "prescription"))  # type: ignore
            edge_types.append(("prescription", "occurs", co_occurrence))  # type: ignore

        # add co-occurrence link edges and weights
        if self.co_occurrence_links:

            self.graph_data["disease", "occurs_link", "disease"].edge_index = self.d2d_co_edges
            self.graph_data[
                "prescription", "occurs_link", "prescription"
            ].edge_index = self.p2p_co_edges
            self.graph_data["disease", "occurs_link", "prescription"].edge_index = self.d2p_co_edges
            self.graph_data["prescription", "occurs_link", "disease"].edge_index = self.p2d_co_edges

            edge_types.append(("disease", "occurs_link", "disease"))  # type: ignore
            edge_types.append(("prescription", "occurs_link", "prescription"))  # type: ignore
            edge_types.append(("disease", "occurs_link", "prescription"))  # type: ignore
            edge_types.append(("prescription", "occurs_link", "disease"))  # type: ignore

            # edge weights
            self.graph_data["disease", "tree", "disease"].edge_weights = nn.Parameter(
                torch.ones(self.disease_edges.shape[1], dtype=torch.float32),
                requires_grad=self.trainable_edge_weights,
            )
            self.graph_data["prescription", "tree", "prescription"].edge_weights = nn.Parameter(
                torch.ones(self.prescription_edges.shape[1], dtype=torch.float32),
                requires_grad=self.trainable_edge_weights,
            )

            if self.co_occurrence_graph:
                self.graph_data[co_occurrence, "occurs", "disease"].edge_weights = nn.Parameter(
                    torch.ones(self.node2d.shape[1], dtype=torch.float32),
                    requires_grad=self.trainable_edge_weights,
                )
                self.graph_data["disease", "occurs", co_occurrence].edge_weights = nn.Parameter(
                    torch.ones(self.d2node.shape[1], dtype=torch.float32),
                    requires_grad=self.trainable_edge_weights,
                )
                self.graph_data[
                    co_occurrence, "occurs", "prescription"
                ].edge_weights = nn.Parameter(
                    torch.ones(self.node2p.shape[1], dtype=torch.float32),
                    requires_grad=self.trainable_edge_weights,
                )
                self.graph_data[
                    "prescription", "occurs", co_occurrence
                ].edge_weights = nn.Parameter(
                    torch.ones(self.p2node.shape[1], dtype=torch.float32),
                    requires_grad=self.trainable_edge_weights,
                )

            if co_link_config.edge_weights:  # type: ignore

                self.graph_data[
                    "disease", "occurs_link", "disease"
                ].edge_weights = self.d2d_co_weights
                self.graph_data[
                    "prescription", "occurs_link", "prescription"
                ].edge_weights = self.p2p_co_weights
                self.graph_data[
                    "disease", "occurs_link", "prescription"
                ].edge_weights = self.d2p_co_weights
                self.graph_data[
                    "prescription", "occurs_link", "disease"
                ].edge_weights = self.p2d_co_weights

            else:

                # we modify edge weights for weighting
                # the contribution of edge types
                # to avoid writing a new operator
                # INTRA
                assert co_link_config is not None  # mypy
                self.graph_data["disease", "occurs_link", "disease"].edge_weights = nn.Parameter(
                    torch.ones(self.d2d_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_intra,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore
                self.graph_data[
                    "prescription", "occurs_link", "prescription"
                ].edge_weights = nn.Parameter(
                    torch.ones(self.p2p_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_intra,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore

                # INTER
                self.graph_data[
                    "disease", "occurs_link", "prescription"
                ].edge_weights = nn.Parameter(
                    torch.ones(self.d2p_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_inter,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore
                self.graph_data[
                    "prescription", "occurs_link", "disease"
                ].edge_weights = nn.Parameter(
                    torch.ones(self.p2d_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_inter,
                    requires_grad=self.trainable_edge_weights,
                )  # type: ignore

        # contraction setup
        contractions_nodes = []
        if contractions_type is not None:
            contractions_nodes = ["disease", "prescription"]

        # get GNN
        self.gnn = OntologyHeteroGNN(
            edge_types=edge_types,
            embedding_dim=embedding_dim,
            graph_num_layers=graph_num_layers,
            graph_num_filters=graph_num_filters,
            convolution_operator=convolution_operator,
            attention_heads=attention_heads,
            divisor=divisor,
            edge_weights=(False if co_link_config is None else co_link_config.edge_weights),
            gbert_mode=gbert_mode,
            contractions=contractions_nodes,
            contractions_type=contractions_type,
        )

        d_len = len(self.disease_idx_mapping)
        p_len = len(self.prescription_idx_mapping)
        total = num_special_tokens + d_len + p_len
        logging.info(
            f"[EMBEDDING] Final embedding size: ({num_special_tokens}, {d_len}, {p_len}) = {total}"
        )

        # additional setup if training with triplet loss
        if co_occurrence is not None and co_occurrence_loss > 0.0 and triplet_loss > 0.0:

            logging.info(
                f"[TRIPLETS] building triplet loss: {triplet_loss}, margin: {triplet_margin}"
            )
            self.triplet_loss_alpha = triplet_loss
            self.triplet_margin = triplet_margin
            self.triplet_loss = nn.TripletMarginLoss(
                margin=triplet_margin,
                p=2.0,
                swap=True,  # swap {a, p} if this makes for a harder negativ
            )

            # build validation triplets
            self.val_triplets = self.triplets_get_batch(self.co_val_ids, split="val", samples=1)
            logging.info(f"[TRIPLETS] validation triplets: {self.val_triplets.shape}")

            # train dataloader
            logging.info(f"[TRIPLETS] batch size: {triplet_batch_size}")
            self.triplet_train_loader = iter(
                InfiniteDataLoader(
                    TensorDataset(self.co_train_ids),
                    shuffle=True,
                    batch_size=triplet_batch_size,
                )
            )

        # Initialize parameters
        self.init_params()

        # Lookup Cache Variable
        self.lookup_cache = None

    def map_embeddings(self, dropout_occurrence: float = 0.0):
        """
        Runs the graph layers on the input embeddings of all nodes

        Parameter
        ---------
        dropout_occurrence: apply dropout to edges towards the occurrence
            nodes (e.g. patients/visits)
        """

        # apply dropout during training
        edge_dict = {}
        if self.training and dropout_occurrence > 0.0:
            for key, edges in self.graph_data.edge_index_dict.items():
                if key[2] == self.co_occurrence:  # dropout if edge target is co-node
                    num_co_nodes = len(self.co_occurrence_voc.word2idx)  # type: ignore
                    edge_dict[key], weights_dict = dropout_adj(
                        edge_index=edges,
                        p=dropout_occurrence,
                        num_nodes=num_co_nodes,
                        training=self.training,
                        force_undirected=False,
                    )

                else:  # leave other edges untouched
                    edge_dict[key] = edges
        else:
            edge_dict = self.graph_data.edge_index_dict

        # map co_occurrence nodes if using static features
        if self.co_occurrence_static:
            node_dict = {
                key: self.node_feature_projector(val) if (key == self.co_occurrence) else val
                for key, val in self.graph_data.x_dict.items()
            }
        else:
            node_dict = self.graph_data.x_dict

        # run the GNN and return
        if self.co_occurrence_links:
            dev = node_dict["disease"].device
            self.graph_data.to(dev)

        return self.gnn(
            node_dict,
            edge_dict,
            None if not self.co_occurrence_links else self.graph_data.edge_weights_dict,
        )

    def get_all_graph_embeddings(self) -> Dict[str, torch.Tensor]:
        """Retrieve dictionary of all graph embedding types"""
        emb = self.map_embeddings()

        if hasattr(self, "co_occurrence") and self.co_occurrence is not None:
            return {
                "disease": emb["disease"],
                "prescription": emb["prescription"],
                self.co_occurrence: emb[self.co_occurrence],
            }
        else:
            return {"disease": emb["disease"], "prescription": emb["prescription"]}

    @torch.no_grad()
    def triplets_get_batch(
        self,
        anchor_ids: Sequence[int],
        split: Optional[str] = None,
        connection_nodes: str = "disease",
        samples: int = 1,
    ) -> torch.Tensor:
        """
        Returns a batch of sampled triplets from a batch of anchor ids

        Parameter
        ---------
        anchor_ids: -
        split: data/node split to consider
        connection_nodes: see `triplets_from_anchor`
        samples: maximum triplets to sample per anchor

        Return
        ------
        torch.Tensor[samples, 3] with columns: anchor, positiv, negativ
        """

        def getter(idx):
            return self.triplets_from_anchor(
                idx, split=split, connection_nodes=connection_nodes, samples=samples
            )

        batch_tris = map(getter, anchor_ids)  # type: ignore
        batch_tris = filter(lambda x: x is not None, batch_tris)  # type: ignore
        batch_tris = torch.cat(list(batch_tris))  # type: ignore

        return batch_tris  # type: ignore

    @torch.no_grad()
    def triplets_from_anchor(
        self,
        anchor_id: int,
        split: Optional[str] = None,
        connection_nodes: str = "disease",
        samples: int = 1,
    ) -> Optional[torch.Tensor]:
        """
        Computes a positive and a negative sample for the given
        anchor co-occurrence node
        Positives share at least one disease 2-hop subgraph, negatives don't share any disease
        Returns None if no triplet could be found within the given split

        Loosely based on:
            https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html#k_hop_subgraph


        Parameter
        ---------
        anchor_id: anchor node id
        split: if given, the split to consider
            will mask out all options not in split
        connection_nodes: the type of node to use
            to connect the occurrence nodes {'disease', 'prescription'}
        samples: number of triplets attempting to extract

        Return
        ------
        torch.Tensor[samples, 3] with columns: anchor, positiv, negativ
        """

        # get number of occurrence ndoes
        num_nodes_co = len(self.graph_data.x_dict[self.co_occurrence])
        num_nodes_con = len(self.graph_data.x_dict[connection_nodes])

        # get edges
        src_co, target_co = self.graph_data.edge_index_dict[
            (self.co_occurrence, "occurs", connection_nodes)
        ]
        src_con, target_con = self.graph_data.edge_index_dict[
            (connection_nodes, "occurs", self.co_occurrence)
        ]

        # get masks
        node_mask_co = src_co.new_empty(num_nodes_co, dtype=torch.bool)
        edge_mask_co = src_co.new_empty(src_co.size(0), dtype=torch.bool)

        node_mask_con = src_co.new_empty(num_nodes_con, dtype=torch.bool)
        edge_mask_con = src_co.new_empty(src_con.size(0), dtype=torch.bool)

        # anchor id on device
        anchor_id = torch.tensor([anchor_id], device=src_co.device)  # type: ignore
        subsets = [anchor_id]

        # initialize masks
        node_mask_co.fill_(False)
        node_mask_con.fill_(False)

        # anchor id
        node_mask_co[anchor_id] = True

        # select connection nodes
        torch.index_select(node_mask_co, 0, src_co, out=edge_mask_con)
        con_node_ids = target_co[edge_mask_con]

        # reveal connected disease/prescr nodes
        node_mask_con[con_node_ids] = True

        # select occurrence nodes
        torch.index_select(node_mask_con, 0, src_con, out=edge_mask_co)
        occurrence_node_ids = target_con[edge_mask_co]

        # create masks
        positive_ids = occurrence_node_ids
        node_mask_co.fill_(False)
        node_mask_co[positive_ids] = True
        positive_mask = node_mask_co
        negative_mask = torch.logical_not(positive_mask)

        # mask splits
        if split is not None:

            # get relevant split on device
            if split == "train":
                self.co_train_mask = self.co_train_mask.to(positive_mask.device)
                split_mask = self.co_train_mask
            else:
                self.co_val_mask = self.co_val_mask.to(positive_mask.device)
                split_mask = self.co_val_mask

            positive_mask = torch.logical_and(positive_mask, split_mask)
            negative_mask = torch.logical_and(negative_mask, split_mask)

        # compute ids
        positive_ids = positive_mask.nonzero().squeeze()
        negative_ids = negative_mask.nonzero().squeeze()

        try:
            samples = min(len(positive_ids), len(negative_ids), samples)
        except TypeError:
            logging.warning(f"[TRIPLET] could not sample any triplets")
            samples = 0

        if samples < 1:
            return None

        # sample and assemble triplets
        positive_ids = positive_ids[torch.randperm(len(positive_ids))[:samples]]
        negative_ids = negative_ids[torch.randperm(len(negative_ids))[:samples]]
        anchor_ids = torch.full(positive_ids.shape, anchor_id.item(), device=positive_ids.device)  # type: ignore

        return torch.stack((anchor_ids, positive_ids, negative_ids)).t()


    def triplet_loss_computation(
        self,
        split: str = "train",
        connection_nodes: str = "disease",
        samples: int = 1,
        graph_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute triplet loss for a training batch or the validation set

        Parameter
        ---------
        split: data/node split to consider, randomly sampled batch
            for `train`, based on a anchor idx dataloader and complete validation
            set for `val`
        connection_nodes: type of nodes to consider for positive nodes for the
            randomly sampled training batch
        samples: number of triplets to sample per anchor
        graph_embeddings: precomputed graph embeddings (differentiable for loss)

        Return
        ------
        Torch.Tensor: computed loss
        """

        # val: pre-create triplets
        # train: get anchor from an index dataloader and sample triplets
        if split == "train":
            batch_anchor_ids = next(self.triplet_train_loader)[0]
            triplets = self.triplets_get_batch(
                batch_anchor_ids,
                split=split,
                samples=samples,
                connection_nodes=connection_nodes,
            )
        else:
            triplets = self.val_triplets

        anchors = triplets[:, 0]
        positives = triplets[:, 1]
        negatives = triplets[:, 2]

        # extract embeddings
        # TODO: investigate dropout
        if graph_embeddings is not None:
            node_embeddings = graph_embeddings[self.co_occurrence]  # type: ignore
        else:
            node_embeddings = self.map_embeddings(dropout_occurrence=0.0)[self.co_occurrence]

        anchors = node_embeddings[anchors]
        positives = node_embeddings[positives]
        negatives = node_embeddings[negatives]

        # compute loss
        loss = self.triplet_loss(anchors, positives, negatives)

        return loss

    def co_occurrence_loss_computation(
        self,
        split: str = "train",
        compute_metrics: bool = False,
        metrics_fast: bool = True,
        batch: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Compute co-occurrence node loss, considers alpha parameter `self.co_occrence_loss`
        of this instance to weight the loss

        Parameter
        ---------
        compute_metrics: also compute metrics
        metrics_fast: compute subset of metrics only
        batch: use a batch of size `self.co_occurrence_batch_size`
            if False, don't batch
        return_embeddings: return the mapped graph embeddings
        """

        assert (
            self.co_occurrence_loss > 0.0
        ), f"loss weight {self.co_occurrence_loss} needs to be > 0.0"

        # get loss
        loss_f = nn.BCEWithLogitsLoss(reduction="mean")

        # get relevant indeces for split
        split_indeces = self.co_train_ids if split == "train" else self.co_val_ids
        device = self.graph_data.x_dict["disease"].device

        if batch:
            split_indeces = next(self.co_train_loader)[0]

        # move required targets
        d_targets = self.co_d_target[split_indeces].to(device, non_blocking=True)
        p_targets = self.co_p_target[split_indeces].to(device, non_blocking=True)

        # compute predictions
        dropout = (
            self.co_occurrence_dropout if split == "train" else 0.0
        )  # dropout only during train
        graph_embeddings = self.map_embeddings(dropout_occurrence=dropout)
        node_embeddings = graph_embeddings[self.co_occurrence][split_indeces]
        node_preds_d = self.co_d_clf(node_embeddings)
        node_preds_p = self.co_p_clf(node_embeddings)

        # compute loss
        d_loss = loss_f(node_preds_d, d_targets)
        p_loss = loss_f(node_preds_p, p_targets)
        loss = self.co_occurrence_loss * (d_loss + p_loss)

        data_dict = {}
        data_dict["loss"] = loss

        if compute_metrics:
            acc_container_d = metric_report(
                node_preds_d.detach().cpu().numpy(),
                d_targets.cpu().numpy(),
                verbose=False,
                fast=metrics_fast,
            )
            for k, v in acc_container_d.items():
                data_dict[f"d_{k}"] = v

            acc_container_p = metric_report(
                node_preds_p.detach().cpu().numpy(),
                p_targets.cpu().numpy(),
                verbose=False,
                fast=metrics_fast,
            )
            for k, v in acc_container_p.items():
                data_dict[f"p_{k}"] = v

        if return_embeddings:
            data_dict["graph_embeddings"] = graph_embeddings

        return data_dict

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Maps `token_ids` to graph learned embeddings
        """

        # get node embeddings
        if use_cache and self.lookup_cache is not None:
            # use cached
            concat_embeddings = self.lookup_cache

        else:

            emb = self.map_embeddings()

            # gather relevant embeddings
            disease_embeddings = emb["disease"][self.disease_idx_mapping]
            prescription_embeddings = emb["prescription"][self.prescription_idx_mapping]

            # compose final embedding matrix
            concat_embeddings = torch.cat(
                [self.special_embedding, disease_embeddings, prescription_embeddings],
                dim=0,
            )

            # cache embeddings
            self.lookup_cache = concat_embeddings  # type: ignore

        # retrieve embeddings
        # return concat_embeddings[token_ids]
        return F.embedding(token_ids, concat_embeddings)

    def init_params(self):
        """Initializes embedding parameters"""
        torch_geometric.nn.inits.glorot(self.disease_embedding)
        torch_geometric.nn.inits.glorot(self.prescription_embedding)
        torch_geometric.nn.inits.glorot(self.special_embedding)  # special tokens [CLS, PAD, ...]

        if self.co_occurrence_graph and not self.co_occurrence_static:
            logging.info("[CO] randomly initialize co node embeddings")
            torch_geometric.nn.inits.glorot(self.co_occurrence_nodes)
        else:
            logging.info("[CO] DONT initialize co node embeddings")
