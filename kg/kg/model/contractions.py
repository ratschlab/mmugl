# ===============================================
#
# Graph Contractions
#
# ===============================================
from typing import Optional, Tuple

import torch
from torch_geometric.utils import coalesce, remove_self_loops, sort_edge_index


def edge_pooling_rewrite_edges(
    edge_index: torch.Tensor,
    clustering: torch.Tensor,
    edge_weights: torch.Tensor = None,
    endpoint_indeces: Tuple[int, ...] = (0,),
    rm_self_loops: bool = False,
    rm_duplicates: bool = False,
    coalesce_reduce="mean",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Rewrites edge indeces using the provided clustering information
    after computing a set of contractions

    Parameter
    ---------
    edge_index: torch.Tensor[2, num_edges]
        the set of edges to be rewritten
    clustering: torch.Tensor
        the clustering information of the associated
        pooling step
    edge_weights: torch.Tensor
        optional edge weight tensor
        to be pruned accordingly
    endpoint_indeces: Tuple[int]
        the indeces of the edge endpoint(s)
        to rewrite, default only 0 i.e. src
    remove_selfloops: bool
        whether to remove selfloops occurring
        after the rewrite
    remove_duplicates: bool
        whether to remove duplicates occurring
        after the rewrite
    coalesce_reduce: str
        mode for the reduction operation
        on the duplicate removal

    Return
    ------
    edge_index: torch.Tensor
        the rewritten edge index
    edge_weights: torch.Tensor
        pruned edge weights, if not
        provided returns `None`
    """

    new_edge_index = torch.clone(edge_index)

    # rewrite
    for enpoint_index in endpoint_indeces:
        endpoints = new_edge_index[enpoint_index, :]  # extract endpoints
        endpoints = clustering[endpoints]  # remap using clustering
        new_edge_index[enpoint_index, :] = endpoints  # rewrite

    # remove self loops after rewrite
    if rm_self_loops:
        new_edge_index, edge_weights = remove_self_loops(new_edge_index, edge_weights)

    # coalesce duplicate edges after rewrite
    if rm_duplicates:
        if edge_weights is None:
            new_edge_index = coalesce(new_edge_index)
        else:
            new_edge_index, edge_weights = coalesce(
                new_edge_index, edge_weights, reduce=coalesce_reduce
            )

    return new_edge_index, edge_weights


def edge_pooling_expand_nodes(nodes: torch.Tensor, clustering: torch.Tensor):
    """
    Computes the expanded node
    embedding matrix given a computed
    clustering after edge contractions

    Parameter
    ---------
    nodes: torch.Tensor
        node embeddings as a result of
        the associated contraction operation
    clustering: torch.Tensor
        clustering information of the
        associated contraction

    Return
    ------
    expaned_nodes: torch.Tensor
        the expanded node embeddings
        considering the given clustering
    """
    expanded_nodes = nodes[clustering]
    return expanded_nodes
