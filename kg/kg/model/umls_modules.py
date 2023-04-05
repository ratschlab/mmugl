# ===============================================
#
# Torch Graph Learning Modules
# to learn from UMLS extracted graphs
#
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
    GraphNorm,
    HeteroConv,
    SAGEConv,
    to_hetero,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj

from kg.data.contants import ALLOWED_CO_OCCURRENCE
from kg.data.datasets import CodeTokenizer
from kg.data.graph import (
    CoLinkConfig,
    Vocabulary,
    build_atc_tree,
    build_co_occurrence_edges,
    build_co_occurrence_vocabulary,
    build_cominbed_edges,
    build_icd9_tree,
    build_stage_one_edges,
    build_stage_two_edges,
    cluster_co_occurrence_nodes,
    load_co_occurrence_features,
)
from kg.data.umls import (
    build_sapbert_node_matrix,
    build_umls_co_occurrence_edges,
    build_umls_co_occurrence_links,
    build_umls_graph_from_networkx,
)
from kg.model.contractions import edge_pooling_expand_nodes, edge_pooling_rewrite_edges
from kg.model.utility_modules import MLP
from kg.training.dataloader import InfiniteDataLoader
from kg.utils.metrics import metric_report


class UMLSGNN(nn.Module):
    """A GNN to be used on an `UMLSGraphEmbedding`"""

    def __init__(
        self,
        edge_types: Sequence[Tuple[str, str, str]],
        embedding_dim: int,
        graph_num_layers: int = 1,
        graph_num_filters: int = 1,
        convolution_operator: str = "SAGEConv",
        attention_heads: int = 1,
        edge_weights: bool = False,
        node_aggregation: str = "sum",
        graph_normalization: bool = False,
        contractions_type: str = None,
    ):
        """
        Constructor for `UMLSGraphEmbedding`

        Parameters
        ----------
        edge_types: A list of tuples describing
            the edge types (to be used in construction with
            a Pytorch Geometric `HeteroConv` module)
        embedding_dim: -
        graph_num_layers: -
        graph_num_filters: int
            number of distinct layer stacks
            aggregate is `max` over the stacks
        convolution_operator: GNN operator to use
        attention_heads: number of attention heads to use
            for some `convolution_operator`, ignored for others.
        edge_weights: bool
            whether edge weights are provided for `occurs_link` edges
            works only with GraphConv
        node_aggregation: str
            how to aggregate contributions of different
            edge types in a heterogenous GNN; default is 'sum'
        graph_normalization: bool
            whether to apply GraphNorm after all but
            last convolution: `https://arxiv.org/pdf/2009.03294.pdf`
        contractions: List[str]
            node types to apply contractions to
        contractions_type: str
            scoring func for contractions
        """
        super(UMLSGNN, self).__init__()

        # set parameters
        self.graph_num_layers = graph_num_layers
        self.graph_num_filters = graph_num_filters
        self.embedding_dim = embedding_dim
        self.convolution_operator = convolution_operator
        self.attention_heads = attention_heads
        self.edge_weights = edge_weights

        # build graph layers
        self.graph_convs = nn.ModuleList()
        for _ in range(graph_num_filters):
            convs = nn.ModuleList()
            for _ in range(graph_num_layers):
                conv = HeteroConv(
                    {edge: self.conv_module_builder(edge) for edge in edge_types},
                    aggr=node_aggregation,
                )
                convs.append(conv)
            self.graph_convs.append(convs)

        # collect node types from edges
        sources = set(map(lambda x: x[0], edge_types))
        dests = set(map(lambda x: x[2], edge_types))
        node_types = sources.union(dests)

        # normalization
        self.graph_normalization = graph_normalization

        if self.graph_normalization:

            self.norm_layers = nn.ModuleList()
            for _ in range(graph_num_filters):
                norms = nn.ModuleList()
                for _ in range(graph_num_layers - 1):
                    normalizers = nn.ModuleDict()
                    for node in node_types:
                        normalizers[node] = GraphNorm(self.embedding_dim)
                    norms.append(normalizers)
                self.norm_layers.append(norms)

            logging.info(f"Using graph normalization: {self.norm_layers}")

        # init contractions
        self.do_contractions = contractions_type is not None
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
                    pool = EdgePooling(
                        self.embedding_dim,
                        scoring_func,
                        dropout=0,
                        add_to_edge_score=0.0,
                    )  # 0.5
                    pool_layers.append(pool)
                self.pool_stacks.append(pool_layers)

            logging.info(f"[{self.__class__.__name__}`] contraction score func: {scoring_func}")
            logging.info(
                f"[{self.__class__.__name__}`] Built contraction layers: {self.pool_stacks}"
            )

        # log build
        logging.info(f"Built: {self}")

    def conv_module_builder(self, edge) -> MessagePassing:

        if self.convolution_operator == "GINConv":

            return GINConv(
                MLP([self.embedding_dim, 2 * self.embedding_dim, self.embedding_dim]),
                train_eps=True,
            )

        elif self.convolution_operator == "GATConv":

            return GATConv(
                in_channels=(self.embedding_dim, self.embedding_dim),
                out_channels=self.embedding_dim,
                concat=False,
                heads=self.attention_heads,
            )

        elif self.convolution_operator == "SAGEConv":

            return SAGEConv(
                in_channels=(self.embedding_dim, self.embedding_dim),
                out_channels=self.embedding_dim,
                aggr="mean",  # max, mean, add, lstm
                normalize=False,
            )

        elif self.convolution_operator == "GraphConv":

            aggr = "mean"
            if self.edge_weights and "occurs_link" in edge[1]:
                aggr = "add"
            logging.debug(f"[GraphConv] Edge {edge}: aggregation: {aggr}")

            return GraphConv(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                aggr=aggr,
            )

        else:
            raise NotImplementedError(f"Non-supported GNN operator {self.convolution_operator}")

    def apply_contraction(
        self,
        pool_layer: EdgePooling,
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
        clustering_info = None

        target_nodes = x_dict["cui"]
        target_edges = edge_index_dict[("cui", "umls", "cui")]
        batch_index = torch.zeros(
            target_nodes.shape[0], dtype=torch.int64, device=target_nodes.device
        )

        contracted_nodes, _, _, unpool_info = pool_layer(target_nodes, target_edges, batch_index)
        clustering_info = unpool_info.cluster

        x_dict["cui"] = contracted_nodes

        # rewrite edges
        for edge in edge_index_dict.keys():

            # get edge weights
            if edge_weights_dict is None:
                edge_weights = None
            else:
                edge_weights = edge_weights_dict[edge]

            # both endpoints affected by pooling
            new_edge_index = None
            if edge[0] == edge[2] and edge[0] == "cui":

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
            elif edge[0] == "cui":

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
            elif edge[2] == "cui":

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

        return x_dict, edge_index_dict, edge_weights_dict, clustering_info

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, ...], torch.Tensor],
        edge_weights: Optional[Dict[Tuple[str, ...], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:

        # get input for each stack of layers
        activations = [x_dict for _ in range(self.graph_num_filters)]

        # go over each stack of layers
        for l, layers in enumerate(self.graph_convs):

            # book-keeping of contractions
            clustering_layers = []

            # single stack of layers
            for i, conv in enumerate(layers):

                # graph conv layer with edge weights
                if edge_weights is None:
                    activations[l] = conv(activations[l], edge_index_dict)
                else:
                    activations[l] = conv(activations[l], edge_index_dict, edge_weights)

                # apply ReLU and Norm, after all but last layer
                if i < (self.graph_num_layers - 1):

                    # Normalize
                    if self.graph_normalization:
                        activations[l] = {
                            key: self.norm_layers[l][i][key](x) for key, x in activations[l].items()
                        }

                    # ReLU
                    activations[l] = {key: x.relu() for key, x in activations[l].items()}

                # pooling / contractions
                if self.do_contractions:
                    (
                        activations[l],
                        edge_index_dict,
                        edge_weights,
                        clustering_info,
                    ) = self.apply_contraction(
                        self.pool_stacks[l][i],
                        activations[l],
                        edge_index_dict,
                        edge_weights,
                    )
                    clustering_layers.append(clustering_info)

                # retrace contractions and expand node embeddings
            if self.do_contractions:
                for clustering in reversed(clustering_layers):

                    # before_shape = activations[l]['cui'].shape
                    activations[l]["cui"] = edge_pooling_expand_nodes(
                        activations[l]["cui"], clustering
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
        return f"`{self.__class__.__name__}` {self.graph_num_layers} layers, {self.graph_num_filters} filters: {self.convolution_operator}:{self.embedding_dim}{heads}"


class UMLSGraphEmbedding(nn.Module):
    """
    Graph Embedding over UMLS nodes
    and co-occurrence nodes (patients/visits)
    """

    def __init__(
        self,
        umls_data: Dict,
        disease_vocabulary: Vocabulary,
        prescription_vocabulary: Vocabulary,
        embedding_dim: int,
        graph_num_layers: int,
        graph_num_filters: int = 1,
        convolution_operator: str = "SAGEConv",
        attention_heads: int = 1,
        num_special_tokens: int = 3,
        data_pd: Optional[pd.DataFrame] = None,
        co_occurrence: Optional[str] = None,
        co_occurrence_subsample: float = 0.3,
        co_occurrence_loss: float = 0.0,
        co_occurrence_dropout: float = 0.0,
        co_occurrence_features: Optional[str] = None,
        co_occurrence_batch_size: int = 16,
        co_occurrence_divisor: int = 1,
        tokenizer: Optional[CodeTokenizer] = None,
        triplet_loss: float = 0.0,
        triplet_batch_size: int = 16,
        triplet_margin: float = 0.1,
        co_link_config: CoLinkConfig = None,
        contractions_type: str = None,
    ):
        """
        Constructor for `UMLSGraphEmbedding`

        Parameter
        ---------
        umls_data: Dict
            graph data
        disease_vocabulary: ICD code vocabulary to use
        prescription_vocabulary: ATC code vocabulary to use
        tokenizer_vocabulary: full vocabulary used for tokenization
        embedding_dim: dimension of the embeddings
        graph_num_layers: number of graph layers to stack
        graph_num_filters: int
            number of distinct layer stacks
            aggregated after last layer
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
        co_occurrence_divisor: for co-occurrence nodes with lower dimensionality
        tokenizer: `CodeTokenizer` for model
        triplet_loss: alpha parameter for the additional triplet loss
        triplet_batch_size: -
        triplet_margin: margin parameter of the triplet loss
        co_link_config: CoLinkConfig
            configuration for co-occurrence links
        contractions_type: str
            perform graph contractions i.e. edge pooling with prov. score func
        """
        super(UMLSGraphEmbedding, self).__init__()

        if co_occurrence_loss > 0.0:
            assert tokenizer is not None, "Need to pass tokenizer if co-occurrence used"
            self.tokenizer = tokenizer

        assert co_occurrence_divisor == 1, "co_occurrence_divisor not supported to be != 1"

        self.co_occurrence_dropout = co_occurrence_dropout
        self.co_occurrence = co_occurrence
        self.co_occurrence_batch_size = co_occurrence_batch_size

        # check link config and conv operator compatibility
        if co_link_config is not None:
            assert_msg = f"Cannot use {convolution_operator} with co links"
            assert convolution_operator in {"GraphConv", "GCNConv"}, assert_msg

        logging.info(f"[UMLS] loaded graph data, hops: {umls_data['grow_hops']}")
        umls_graph = umls_data["graph"]
        icd9_to_cui_map = umls_data["icd9_to_cui_map"]
        atc_to_cui_map = umls_data["atc_to_cui_map"]

        # Check if SapBert embeddings are available
        if "sapbert" in list(umls_graph.nodes(data=True))[0][1].keys():
            logging.info(f"[GRAPH] using SapBert embeddings as initialization")
            self.sapbert = True
        else:
            self.sapbert = False

        # build PyG graph resources
        cui_vocab, cui_edges = build_umls_graph_from_networkx(
            umls_graph, vocabulary=umls_data["tokenizer"].vocabulary
        )
        self.cui_vocab = cui_vocab
        self.cui_edges = nn.Parameter(torch.tensor(cui_edges), requires_grad=False)

        # initialize the node representations
        num_cui_nodes = len(cui_vocab.word2idx)

        # initialize with SapBERT or random trainable
        if self.sapbert:
            self.cui_embedding = build_sapbert_node_matrix(umls_graph, cui_vocab)
            self.cui_feature_projector = nn.Linear(self.cui_embedding.shape[1], int(embedding_dim))
        else:
            self.cui_embedding = nn.Parameter(torch.Tensor(num_cui_nodes, embedding_dim))
        logging.info(f"{self.__class__.__name__} has {num_cui_nodes} nodes")

        # initialize the embeddings for special tokens
        self.special_embedding = nn.Parameter(torch.Tensor(num_special_tokens, embedding_dim))

        # Co-occurrence
        self.co_occurrence_voc = None
        if data_pd is not None and co_occurrence is not None:
            logging.info("[GRAPH] building co-occurrence graph")
            self.co_occurrence_graph = True

            # build node vocabulary
            self.co_occurrence_voc = build_co_occurrence_vocabulary(
                data_pd, co_occurrence, subsample=co_occurrence_subsample
            )

            # build edges
            edges = build_umls_co_occurrence_edges(
                data_pd,
                self.co_occurrence_voc,
                self.cui_vocab,
                icd9_to_cui_map,
                atc_to_cui_map,
                co_occurrence,
                co_occurrence_loss=co_occurrence_loss,
                tokenizer=tokenizer,
            )

            self.cui2node = nn.Parameter(torch.tensor(edges["cui2node"]), requires_grad=False)
            self.node2cui = nn.Parameter(torch.tensor(edges["node2cui"]), requires_grad=False)

            # build co-occurrence node embeddings
            num_nodes = len(self.co_occurrence_voc.word2idx)
            if co_occurrence_features is not None:  # build from features
                self.co_occurrence_static = True

                # get static node features
                self.co_occurrence_nodes = load_co_occurrence_features(
                    co_occurrence_features,
                    self.co_occurrence_voc,
                    data_pd,
                    co_occurrence,
                )

                # additional projection layer to go from static feature
                # space into graph embedding dimensional space
                self.node_feature_projector = nn.Linear(
                    self.co_occurrence_nodes.shape[1], int(embedding_dim)
                )

                logging.info(
                    f"[GRAPH] co-occurrence nodes (features, static): {self.co_occurrence_nodes.shape}"
                )

            else:  # trainable embeddings
                self.co_occurrence_static = False
                self.co_occurrence_nodes = nn.Parameter(torch.Tensor(num_nodes, int(embedding_dim)))
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
                self.co_d_clf = MLP([int(embedding_dim)] + list(repeat(128, 2)) + [d_nodes])
                self.co_p_clf = MLP([int(embedding_dim)] + list(repeat(128, 2)) + [p_nodes])

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

            # build links
            edge_data = build_umls_co_occurrence_links(
                data_pd,
                self.cui_vocab,
                icd9_to_cui_map,
                atc_to_cui_map,
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
                    requires_grad=False,
                )  # type: ignore
                self.p2p_co_weights = nn.Parameter(
                    edge_data["p2p_weights"] * co_link_config.alpha_intra,
                    requires_grad=False,
                )  # type: ignore
                self.d2p_co_weights = nn.Parameter(
                    edge_data["d2p_weights"] * co_link_config.alpha_inter,
                    requires_grad=False,
                )  # type: ignore
                self.p2d_co_weights = nn.Parameter(
                    edge_data["p2d_weights"] * co_link_config.alpha_inter,
                    requires_grad=False,
                )  # type: ignore

        # assemble heterogeneous graph data
        self.graph_data = HeteroData()
        self.graph_data["cui"].x = self.cui_embedding
        self.graph_data["cui", "umls", "cui"].edge_index = self.cui_edges

        # gather edge types
        edge_types = [("cui", "umls", "cui")]

        # add co-occurrence data components
        if self.co_occurrence_graph:

            # add nodes
            self.graph_data[co_occurrence].x = self.co_occurrence_nodes

            # add edges in all directions
            self.graph_data[co_occurrence, "occurs", "cui"].edge_index = self.node2cui
            self.graph_data["cui", "occurs", co_occurrence].edge_index = self.cui2node

            # add additional edge types
            edge_types.append((co_occurrence, "occurs", "cui"))  # type: ignore
            edge_types.append(("cui", "occurs", co_occurrence))  # type: ignore

        # add co-occurrence link edges and weights
        if self.co_occurrence_links:

            # set edge indeces
            self.graph_data["cui", "occurs_link_d2d", "cui"].edge_index = self.d2d_co_edges
            self.graph_data["cui", "occurs_link_p2p", "cui"].edge_index = self.p2p_co_edges
            self.graph_data["cui", "occurs_link_d2p", "cui"].edge_index = self.d2p_co_edges
            self.graph_data["cui", "occurs_link_p2d", "cui"].edge_index = self.p2d_co_edges

            # gather edge types
            edge_types.append(("cui", "occurs_link_d2d", "cui"))  # type: ignore
            edge_types.append(("cui", "occurs_link_p2p", "cui"))  # type: ignore
            edge_types.append(("cui", "occurs_link_d2p", "cui"))  # type: ignore
            edge_types.append(("cui", "occurs_link_p2d", "cui"))  # type: ignore

            # edge weights for UMLs edges
            self.graph_data["cui", "umls", "cui"].edge_weights = nn.Parameter(
                torch.ones(self.cui_edges.shape[1], dtype=torch.float32),
                requires_grad=False,
            )

            # edge weights for Co-Occurrence Nodes (Patients/Visits)
            if self.co_occurrence_graph:
                self.graph_data[co_occurrence, "occurs", "cui"].edge_weights = nn.Parameter(
                    torch.ones(self.node2cui.shape[1], dtype=torch.float32),
                    requires_grad=False,
                )
                self.graph_data["cui", "occurs", co_occurrence].edge_weights = nn.Parameter(
                    torch.ones(self.cui2node.shape[1], dtype=torch.float32),
                    requires_grad=False,
                )

            # set edge weights
            if co_link_config.edge_weights:  # type: ignore
                self.graph_data["cui", "occurs_link_d2d", "cui"].edge_weights = self.d2d_co_weights
                self.graph_data["cui", "occurs_link_p2p", "cui"].edge_weights = self.p2p_co_weights
                self.graph_data["cui", "occurs_link_d2p", "cui"].edge_weights = self.d2p_co_weights
                self.graph_data["cui", "occurs_link_p2d", "cui"].edge_weights = self.p2d_co_weights

            else:
                # we modify edge weights for weighting
                # the contribution of edge types
                # to avoid writing a new operator

                # INTRA
                assert co_link_config is not None  # mypy
                self.graph_data["cui", "occurs_link_d2d", "cui"].edge_weights = nn.Parameter(
                    torch.ones(self.d2d_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_intra,
                    requires_grad=False,
                )  # type: ignore
                self.graph_data["cui", "occurs_link_p2p", "cui"].edge_weights = nn.Parameter(
                    torch.ones(self.p2p_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_intra,
                    requires_grad=False,
                )  # type: ignore

                # INTER
                self.graph_data["cui", "occurs_link_d2p", "cui"].edge_weights = nn.Parameter(
                    torch.ones(self.d2p_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_inter,
                    requires_grad=False,
                )  # type: ignore
                self.graph_data["cui", "occurs_link_p2d", "cui"].edge_weights = nn.Parameter(
                    torch.ones(self.p2d_co_edges.shape[1], dtype=torch.float32)
                    * co_link_config.alpha_inter,
                    requires_grad=False,
                )  # type: ignore

        # get GNN
        node_aggregation = "mean" if self.co_occurrence_links else "sum"
        logging.info(f"[GRAPH] Node aggregation: {node_aggregation}")

        self.gnn = UMLSGNN(
            edge_types=edge_types,
            embedding_dim=embedding_dim,
            graph_num_layers=graph_num_layers,
            graph_num_filters=graph_num_filters,
            convolution_operator=convolution_operator,
            attention_heads=attention_heads,
            edge_weights=(False if co_link_config is None else co_link_config.edge_weights),
            node_aggregation=node_aggregation,
            graph_normalization=self.co_occurrence_links,
            contractions_type=contractions_type,
        )

        cui_len = len(cui_vocab.idx2word)
        total = cui_len
        logging.info(
            f"[EMBEDDING] Final embedding size: ({num_special_tokens}, {cui_len-num_special_tokens}) = {total}"
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
                    edge_dict[key], _ = dropout_adj(
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

        # Project SapBert embeddings
        if self.sapbert:
            node_dict["cui"] = self.cui_feature_projector(node_dict["cui"])

        # run the GNN and return
        if self.co_occurrence_links:
            dev = node_dict["cui"].device
            self.graph_data.to(dev)

        # run the GNN and return
        return self.gnn(
            node_dict,
            edge_dict,
            None if not self.co_occurrence_links else self.graph_data.edge_weights_dict,
        )

    def get_all_graph_embeddings(self) -> Dict[str, torch.Tensor]:
        """Retrieve dictionary of all graph embedding types"""
        emb = self.map_embeddings()

        if hasattr(self, "co_occurrence") and self.co_occurrence is not None:
            return {"cui": emb["cui"], self.co_occurrence: emb[self.co_occurrence]}
        else:
            return {"cui": emb["cui"]}

    @torch.no_grad()
    def triplets_get_batch(
        self, anchor_ids: Sequence[int], split: Optional[str] = None, samples: int = 1
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
            return self.triplets_from_anchor(idx, split=split, samples=samples)

        batch_tris = map(getter, anchor_ids)  # type: ignore
        batch_tris = filter(lambda x: x is not None, batch_tris)  # type: ignore
        batch_tris = torch.cat(list(batch_tris))  # type: ignore

        return batch_tris  # type: ignore

    @torch.no_grad()
    def triplets_from_anchor(
        self, anchor_id: int, split: Optional[str] = None, samples: int = 1
    ) -> Optional[torch.Tensor]:
        """
        Computes a positive and a negative sample for the given
        anchor co-occurrence node
        Positives share at least one disease 2-hop subgraph, negatives don't share any CUI
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
        num_nodes_con = len(self.graph_data.x_dict["cui"])

        # get edges
        src_co, target_co = self.graph_data.edge_index_dict[(self.co_occurrence, "occurs", "cui")]
        src_con, target_con = self.graph_data.edge_index_dict[("cui", "occurs", self.co_occurrence)]

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

        # reveal connected cui nodes
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
        samples: int = 1,
        connection_nodes: str = "cui",
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

        connection_nodes = "cui"

        # val: pre-create triplets
        # train: get anchor from an index dataloader and sample triplets
        if split == "train":
            batch_anchor_ids = next(self.triplet_train_loader)[0]
            triplets = self.triplets_get_batch(
                batch_anchor_ids,
                split=split,
                samples=samples,
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
        # loss = self.triplet_loss_alpha * loss

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
        device = self.graph_data.x_dict["cui"].device

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
            cui_embeddings = emb["cui"][3:]  # first three are special (trainable) embeddings

            # compose final embedding matrix
            concat_embeddings = torch.cat([self.special_embedding, cui_embeddings], dim=0)

            # cache embeddings
            self.lookup_cache = concat_embeddings  # type: ignore

        # retrieve embeddings
        return F.embedding(token_ids, concat_embeddings)

    def init_params(self):
        """Initializes embedding parameters"""

        torch_geometric.nn.inits.glorot(self.special_embedding)  # special tokens [CLS, PAD, ...]

        if not self.sapbert:
            logging.info(f"[INIT] Randomly initialize cui node embeddings")
            torch_geometric.nn.inits.glorot(self.cui_embedding)

        if self.co_occurrence_graph and not self.co_occurrence_static:
            logging.info("[INIT] randomly initialize co node embeddings")
            torch_geometric.nn.inits.glorot(self.co_occurrence_nodes)
        else:
            logging.info("[INIT] DONT initialize co node embeddings")
