# ===============================================
#
# Graph Construction Methods
#
# Some methods have been adapted from: https://github.com/jshang123/G-Bert
# ===============================================
import logging
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from itertools import chain, product
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture

from kg.data.contants import ALLOWED_CO_OCCURRENCE
from kg.data.vocabulary import CodeTokenizer, Vocabulary


# ===============================================
#
# Utility
#
# ===============================================
def _remove_duplicate(inputs: Sequence[Any]) -> List[Any]:
    """
    Remove duplicates by set creation

    Source: https://github.com/jshang123/G-Bert
    """
    return list(set(inputs))


def transpose_edges(input_edges: List[Any], remove_duplicates: bool = True) -> List[List[int]]:
    """
    Transposes a 2d edge matrix from list of edges to a 2-sized list
    of lists each containing source and target nodes respectively

    Parameter
    ---------
    input: list of edge tuples
    remove_duplicates: -

    Returns
    -------
    source and target node lists
    """

    if remove_duplicates:
        input_edges = _remove_duplicate(input_edges)
    row = list(map(lambda x: x[0], input_edges))
    col = list(map(lambda x: x[1], input_edges))
    return [row, col]


def build_stage_one_edges(
    tree_mapping: Sequence[Sequence[Any]], graph_voc: Vocabulary
) -> List[List[int]]:
    """
    Builds a directed sparse node-node mapping over
    the given ICD9 code tree given by `tree_mapping`.
    This method builds edges from children to parents.

    Parameter
    ---------
    tree_mapping: the leaf node paths of the tree
    graph_voc: the vocabulary and index mapping of the tree nodes

    Returns
    -------
    A node-node mapping (sparse adj. matrix)
    edge_idx [[1,2,3],[0,1,0]]

    Source: https://github.com/jshang123/G-Bert
    """
    edge_idx = []
    for sample in tree_mapping:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        for i in range(len(sample_idx) - 1):
            # only direct children -> ancestor
            edge_idx.append((sample_idx[i + 1], sample_idx[i]))

    return transpose_edges(edge_idx)


def build_stage_two_edges(
    tree_mapping: Sequence[Sequence[Any]], graph_voc: Vocabulary
) -> List[List[int]]:
    """
    Builds a directed sparse node-node mapping over
    the given ICD9 code tree given by `tree_mapping`.
    This method builds edges from parents to children.

    Parameter
    ---------
    tree_mapping: the leaf node paths of the tree
    graph_voc: the vocabulary and index mapping of the tree nodes

    Returns
    -------
    A node-node mapping (sparse adj. matrix)
    edge_idx [[1,2,3],[0,1,0]]

    Source: https://github.com/jshang123/G-Bert
    """
    edge_idx = []
    for sample in tree_mapping:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        # only ancestors -> leaf node
        edge_idx.extend([(sample_idx[0], sample_idx[i]) for i in range(1, len(sample_idx))])

    return transpose_edges(edge_idx)


def build_cominbed_edges(
    tree_mapping: Sequence[Sequence[Any]], graph_voc: Vocabulary
) -> List[List[int]]:
    """
    Build unidrected graph edges over given tree
    mapping hierarchy from `tree_mapping` and
    the index mapping given by `graph_voc`

    Parameter
    ---------
    tree_mapping: the leaf node paths of the tree
    graph_voc: the vocabulary and index mapping of the tree nodes

    Returns
    -------
    A node-node mapping (sparse adj. matrix)
    edge_idx [[1,2,3],[0,1,0]]

    Source: https://github.com/jshang123/G-Bert
    """
    edge_idx = []
    for sample in tree_mapping:
        sample_idx = list(map(lambda x: graph_voc.word2idx[x], sample))
        for i in range(len(sample_idx) - 1):
            # ancestor <- direct children
            edge_idx.append((sample_idx[i + 1], sample_idx[i]))

            # ancestors -> leaf node
            edge_idx.extend([(sample_idx[0], sample_idx[i]) for i in range(1, len(sample_idx))])

    return transpose_edges(edge_idx)


# ===============================================
#
# ICD-9 Code Hierarchy
#
# ===============================================
def expand_level2():
    """
    create dictionary of level 2 mappings to code ranges
    """
    level2 = [
        "001-009",
        "010-018",
        "020-027",
        "030-041",
        "042",
        "045-049",
        "050-059",
        "060-066",
        "070-079",
        "080-088",
        "090-099",
        "100-104",
        "110-118",
        "120-129",
        "130-136",
        "137-139",
        "140-149",
        "150-159",
        "160-165",
        "170-176",
        "176",
        "179-189",
        "190-199",
        "200-208",
        "209",
        "210-229",
        "230-234",
        "235-238",
        "239",
        "240-246",
        "249-259",
        "260-269",
        "270-279",
        "280-289",
        "290-294",
        "295-299",
        "300-316",
        "317-319",
        "320-327",
        "330-337",
        "338",
        "339",
        "340-349",
        "350-359",
        "360-379",
        "380-389",
        "390-392",
        "393-398",
        "401-405",
        "410-414",
        "415-417",
        "420-429",
        "430-438",
        "440-449",
        "451-459",
        "460-466",
        "470-478",
        "480-488",
        "490-496",
        "500-508",
        "510-519",
        "520-529",
        "530-539",
        "540-543",
        "550-553",
        "555-558",
        "560-569",
        "570-579",
        "580-589",
        "590-599",
        "600-608",
        "610-611",
        "614-616",
        "617-629",
        "630-639",
        "640-649",
        "650-659",
        "660-669",
        "670-677",
        "678-679",
        "680-686",
        "690-698",
        "700-709",
        "710-719",
        "720-724",
        "725-729",
        "730-739",
        "740-759",
        "760-763",
        "764-779",
        "780-789",
        "790-796",
        "797-799",
        "800-804",
        "805-809",
        "810-819",
        "820-829",
        "830-839",
        "840-848",
        "850-854",
        "860-869",
        "870-879",
        "880-887",
        "890-897",
        "900-904",
        "905-909",
        "910-919",
        "920-924",
        "925-929",
        "930-939",
        "940-949",
        "950-957",
        "958-959",
        "960-979",
        "980-989",
        "990-995",
        "996-999",
        "V01-V91",
        "V01-V09",
        "V10-V19",
        "V20-V29",
        "V30-V39",
        "V40-V49",
        "V50-V59",
        "V60-V69",
        "V70-V82",
        "V83-V84",
        "V85",
        "V86",
        "V87",
        "V88",
        "V89",
        "V90",
        "V91",
        "E000-E899",
        "E000",
        "E001-E030",
        "E800-E807",
        "E810-E819",
        "E820-E825",
        "E826-E829",
        "E830-E838",
        "E840-E845",
        "E846-E849",
        "E850-E858",
        "E860-E869",
        "E870-E876",
        "E878-E879",
        "E880-E888",
        "E890-E899",
        "E900-E909",
        "E910-E915",
        "E916-E928",
        "E929",
        "E930-E949",
        "E950-E959",
        "E960-E969",
        "E970-E978",
        "E980-E989",
        "E990-E999",
    ]

    level2_expand = {}
    for i in level2:
        tokens = i.split("-")
        if i[0] == "V":
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["V%02d" % j] = i
        elif i[0] == "E":
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    level2_expand["E%03d" % j] = i
        else:
            if len(tokens) == 1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0]), int(tokens[1]) + 1):
                    level2_expand["%03d" % j] = i
    return level2_expand


def build_icd9_tree(
    unique_codes: Sequence[str],
) -> Tuple[Sequence[Sequence[Any]], Vocabulary]:
    """
    Builds a tree from a set of ICD9 codes according
    to the ICD9 code hierarchy

    Parameters
    ----------
    unique_codes: the code set to create the tree from

    Returns
    -------
    tree_mapping: a tree mapping i.e. a list of lists, where each inner list
                    contains a path from a leaf node to the root
    graph_voc: the vocabulary of the graph/tree with its mappings to indeces
    """

    tree_mapping = []
    graph_voc = Vocabulary()

    root_node = "icd9_root"
    level3_dict = expand_level2()
    for code in unique_codes:

        # the leaf is the code itself
        level1 = code

        # second level is extracted from the code
        # essentially stripping the part after the dot
        # in conventional ICD9 notation
        level2 = level1[:4] if level1[0] == "E" else level1[:3]

        # level three captures the ICD9 code ranges
        level3 = level3_dict[level2]

        # all paths end in the root
        level4 = root_node

        # create path and add to output
        sample = [level1, level2, level3, level4]
        graph_voc.add_sentence(sample)
        tree_mapping.append(sample)

    return tree_mapping, graph_voc


# ===============================================
#
# ATC Prescription Codes
#
# ===============================================
def build_atc_tree(
    unique_codes: Sequence[str],
) -> Tuple[Sequence[Sequence[Any]], Vocabulary]:
    """
    Builds a tree from a set of ATC4 codes according
    to the ATC code hierarchy
    ATC codes: https://www.whocc.no/atc/structure_and_principles/

    Parameters
    ----------
    unique_codes: the code set to create the tree from

    Returns
    -------
    tree_mapping: a tree mapping i.e. a list of lists, where each inner list
                    contains a path from a leaf node to the root
    graph_voc: the vocabulary of the graph/tree with its mappings to indeces
    """
    tree_mapping = []
    graph_voc = Vocabulary()

    root_node = "atc_root"
    for code in unique_codes:

        # ATC codes are nicely structured
        sample = [code]  # add leaf
        sample += [code[:i] for i in [4, 3, 1]]  # add levels by taking slices
        sample += [root_node]  # add root

        # add all nodes (also inner nodes) to graph vocabulary
        graph_voc.add_sentence(sample)
        tree_mapping.append(sample)

    return tree_mapping, graph_voc


# ===============================================
#
# Co-Occurrence Nodes
#
# ===============================================
def build_co_occurrence_vocabulary(
    data_pd: pd.DataFrame,
    co_occurrence_node: str = "patient",
    subsample: Optional[float] = None,
) -> Vocabulary:
    """
    Build `Vocabulary` of co-occurrence nodes

    Parameter
    ---------
    data_pd: DataFrame with columns {SUBJECT_ID, HADM_ID}
    co_occurrence_node: nodes to add for co-occurrence, can be {patient, visit}
    subsample: [0, 1] fraction of nodes to use

    Return
    ------
    `Vocabulary` of nodes
    """

    # build set of nodes
    if co_occurrence_node == "patient":
        nodes = set(f"p-{p}" for p in data_pd["SUBJECT_ID"].unique())

    elif co_occurrence_node == "visit":
        nodes = set(f"v-{v}" for v in data_pd["HADM_ID"].unique())

    else:
        # check co-occurrence nodes type
        assert_msg = (
            f"Co-occurrence nodes needs to be in {ALLOWED_CO_OCCURRENCE}, is {co_occurrence_node}"
        )
        assert co_occurrence_node in ALLOWED_CO_OCCURRENCE, assert_msg

    # init vocabulary
    if subsample is not None:
        logging.info(f"Subsampling co-occurrence nodes to {subsample}")
        nodes = random.sample(list(nodes), int(len(nodes) * subsample))  # type: ignore

    node_vocabulary = Vocabulary()
    node_vocabulary.add_sentence(nodes)

    logging.info(f"Build co-occurrence vocabulary: {len(node_vocabulary.word2idx)}")

    return node_vocabulary


def build_co_occurrence_nodes_df(
    data_pd: pd.DataFrame,
    co_occurrence_node: str = "patient",
):
    """Extract node data based on node type"""

    # build set of nodes
    if co_occurrence_node == "patient":
        prefix = "p-"
        patient_data = data_pd[["SUBJECT_ID", "ATC4", "ICD9_CODE"]].copy()

        # group visits of same patients
        disease_codes = pd.DataFrame(
            patient_data.groupby(["SUBJECT_ID"])["ICD9_CODE"]
            .apply(list)
            .transform(lambda x: list(chain.from_iterable(x)))
        )
        prescription_codes = pd.DataFrame(
            patient_data.groupby(["SUBJECT_ID"])["ATC4"]
            .apply(list)
            .transform(lambda x: list(chain.from_iterable(x)))
        )

        # transform
        node_data = disease_codes.join(prescription_codes, on="SUBJECT_ID").reset_index(drop=False)
        node_data["ID"] = node_data["SUBJECT_ID"].transform(lambda x: f"{prefix}{x}")
        node_data = node_data.drop(labels="SUBJECT_ID", axis=1)

    elif co_occurrence_node == "visit":
        prefix = "v-"
        node_data = data_pd[["HADM_ID", "ATC4", "ICD9_CODE"]].copy()

        # transform
        node_data["ID"] = node_data["HADM_ID"].transform(lambda x: f"{prefix}{x}")
        node_data = node_data.drop(labels="HADM_ID", axis=1).reset_index(drop=True)

    else:
        # check co-occurrence nodes type
        assert_msg = (
            f"Co-occurrence nodes needs to be in {ALLOWED_CO_OCCURRENCE}, is {co_occurrence_node}"
        )
        assert co_occurrence_node in ALLOWED_CO_OCCURRENCE, assert_msg

    return node_data


def build_co_occurrence_edges(
    data_pd: pd.DataFrame,
    co_occurrence_vocabulary: Vocabulary,
    disease_vocabulary: Vocabulary,
    prescription_vocabulary: Vocabulary,
    co_occurrence_node: str = "patient",
    co_occurrence_loss: float = 0.0,
    tokenizer: CodeTokenizer = None,
) -> Dict:
    """
    Builds co-occurrence graph edges based on a preprocessed
    set of patients from MIMIC-III

    Parameter
    ---------
    data_pd: DataFrame with columns {SUBJECT_ID, HADM_ID, ATC4, ICD9_CODE}
    co_occurrence_vocabulary: `Vocabulary` with the co-occurrence nodes
    disease_vocabulary: `Vocabulary` holding the node ids for ICD9 codes
    prescription_vocabulary: `Vocabulary` holding the node ids for ATC4 codes
    co_occurrence_node: nodes to add for co-occurrence, can be {patient, visit}
    co_occurrence_loss: alpha parameter for the co-occurrence auto-encoder loss
        if >0 then the targets are generated and returned
    tokenizer: `CodeTokenizer` for model

    Return
    ------
    List of edges according to Pytorch Geometric format
    """

    if co_occurrence_loss > 0.0:
        assert tokenizer is not None, "Need to pass tokenizer if co-occurrence targets are init."

    # build set of nodes
    node_data = build_co_occurrence_nodes_df(data_pd, co_occurrence_node)
    logging.info(f"Extracted {len(node_data)} potential co-occurrence nodes: {co_occurrence_node}")

    # build edges
    disease2node_edge_idx = []
    node2disease_edge_idx = []
    prescription2node_edge_idx = []
    node2prescription_edge_idx = []

    # initialize co-occurrence targets tensors
    if co_occurrence_loss > 0.0:
        num_nodes = len(co_occurrence_vocabulary.word2idx)
        d_nodes = len(tokenizer.disease_vocabulary.word2idx)  # type: ignore
        p_nodes = len(tokenizer.prescription_vocabulary.word2idx)  # type: ignore
        co_occurrence_target_d = torch.zeros((num_nodes, d_nodes))
        co_occurrence_target_p = torch.zeros((num_nodes, p_nodes))

    for _, row in node_data.iterrows():

        # can be patient or visit of a patient depending on mode
        patient_id = row["ID"]

        # skip if not in vocabulary
        if patient_id not in co_occurrence_vocabulary.word2idx.keys():
            continue

        node = co_occurrence_vocabulary.word2idx[patient_id]  # get node id

        # build edges to disease nodes, undirected
        disease_codes = row["ICD9_CODE"]
        node2disease_edge_idx.extend(
            [(node, disease_vocabulary.word2idx[d]) for d in disease_codes]
        )
        disease2node_edge_idx.extend(
            [(disease_vocabulary.word2idx[d], node) for d in disease_codes]
        )

        # build edges to prescription nodes, undirected
        prescription_codes = row["ATC4"]
        node2prescription_edge_idx.extend(
            [(node, prescription_vocabulary.word2idx[p]) for p in prescription_codes]
        )
        prescription2node_edge_idx.extend(
            [(prescription_vocabulary.word2idx[p], node) for p in prescription_codes]
        )

        # load co-occurrence targets
        if co_occurrence_loss > 0.0:
            patient_index = co_occurrence_vocabulary.word2idx[patient_id]
            d_indeces = [
                tokenizer.disease_vocabulary.word2idx[code]  # type: ignore
                for code in disease_codes
            ]
            co_occurrence_target_d[patient_index, d_indeces] = 1

            p_indeces = [
                tokenizer.prescription_vocabulary.word2idx[code]  # type: ignore
                for code in prescription_codes
            ]
            co_occurrence_target_p[patient_index, p_indeces] = 1

    logging.info(f"Extracted {len(node2disease_edge_idx)} node<->disease co-occurrence edges")
    logging.info(
        f"Extracted {len(node2prescription_edge_idx)} node<->prescription co-occurrence edges"
    )

    node2disease_edge_idx = transpose_edges(node2disease_edge_idx, remove_duplicates=False)  # type: ignore
    disease2node_edge_idx = transpose_edges(disease2node_edge_idx, remove_duplicates=False)  # type: ignore
    node2prescription_edge_idx = transpose_edges(node2prescription_edge_idx, remove_duplicates=False)  # type: ignore
    prescription2node_edge_idx = transpose_edges(prescription2node_edge_idx, remove_duplicates=False)  # type: ignore

    data_dict = {
        "node2disease": node2disease_edge_idx,
        "disease2node": disease2node_edge_idx,
        "prescription2node": prescription2node_edge_idx,
        "node2prescription": node2prescription_edge_idx,
    }

    if co_occurrence_loss > 0.0:
        data_dict["d_targets"] = co_occurrence_target_d  # type: ignore
        data_dict["p_targets"] = co_occurrence_target_p  # type: ignore

        d_hash = co_occurrence_target_d.sum()
        p_hash = co_occurrence_target_p.sum()
        logging.info(f"Extracted co-occurrence targets d: {co_occurrence_target_d.shape}:{d_hash}")
        logging.info(f"Extracted co-occurrence targets p: {co_occurrence_target_p.shape}:{p_hash}")

    return data_dict


def cluster_co_occurrence_nodes(
    data_pd: pd.DataFrame,
    co_occurrence_vocabulary: Vocabulary,
    tokenizer: CodeTokenizer,
    co_occurrence_node: str = "patient",
    max_num_nodes: int = 1000,
):
    """
    Creates a smaller co-occurrence vocabulary of at most `max_num_nodes`
    size by applying clustering

    Parameter
    ---------
    data_pd: the dataset
    co_occurrence_vocabulary: the unreduced co vocabulary
    tokenizer: -
    co_occurrence_node: type of co node
    max_num_nodes: max size of co vocabulary
    """

    node_data = build_co_occurrence_nodes_df(data_pd, co_occurrence_node)

    num_nodes = len(co_occurrence_vocabulary.word2idx)
    d_nodes = len(tokenizer.disease_vocabulary.word2idx)  # type: ignore
    p_nodes = len(tokenizer.prescription_vocabulary.word2idx)  # type: ignore
    co_occurrence_target_d = np.zeros((num_nodes, d_nodes))
    co_occurrence_target_p = np.zeros((num_nodes, p_nodes))
    patient_ids = []

    # collect the relevant targets
    for _, row in node_data.iterrows():

        # can be patient or visit of a patient depending on mode
        patient_id = row["ID"]

        # skip if not in vocabulary
        if patient_id not in co_occurrence_vocabulary.word2idx.keys():
            continue

        # collect patient ids
        patient_ids.append(patient_id)

        # load co-occurrence targets
        disease_codes = row["ICD9_CODE"]
        patient_index = co_occurrence_vocabulary.word2idx[patient_id]
        d_indeces = [
            tokenizer.disease_vocabulary.word2idx[code] for code in disease_codes  # type: ignore
        ]
        co_occurrence_target_d[patient_index, d_indeces] = 1

        prescription_codes = row["ATC4"]
        p_indeces = [
            tokenizer.prescription_vocabulary.word2idx[code]  # type: ignore
            for code in prescription_codes
        ]
        co_occurrence_target_p[patient_index, p_indeces] = 1

    # perform clustering
    full_features = np.concatenate((co_occurrence_target_d, co_occurrence_target_p), axis=1)
    # full_features = co_occurrence_target_d
    logging.info(
        f"[GRAPH] perform clustering to reduce co nodes on features: {full_features.shape}"
    )

    clustering_algo = MiniBatchKMeans(
        n_clusters=max_num_nodes,
        batch_size=8192,
        verbose=1,
        random_state=42,
        init="random",  # k-means++
        n_init=3,
    )

    cluster_labels = clustering_algo.fit_predict(full_features)

    num_clusters = len(set(cluster_labels))
    logging.info(f"[GRAPH] found {num_clusters} clusters")

    keep_patients = []
    label_set = set()
    for label, patient_id in zip(cluster_labels, patient_ids):

        if label in label_set:
            continue

        label_set.add(label)
        keep_patients.append(patient_id)

    # create new vocabulary
    node_vocabulary = Vocabulary()
    node_vocabulary.add_sentence(keep_patients)
    logging.info(f"Build co-occurrence vocabulary: {len(node_vocabulary.word2idx)}")

    return node_vocabulary


def load_co_occurrence_features(
    feature_file: str, voc: Vocabulary, data_pd: pd.DataFrame, co_occurrence: str
) -> torch.Tensor:
    """
    Loads static co_occurrence_features and returns a tensor
    with the initialized features

    Parameter
    ---------
    feature_file: pd.DataFrame as pickle containing static node features
    voc: the vocabulary for the nodes with the node-id to subj.-id mapping
    data_pd: dataset dataframe, used for subject_id to hadm_id mapping

    Return
    ------
    Torch Tensor containing the node features; we use a nn.Parameter to ensure
    the tensor is carried along with the model (especially when used together with
    PyG and PIL, encountered issues with `register_buffer`)
    """

    logging.info(f"Loading static node features from: {feature_file}")

    feature_df = pd.read_pickle(feature_file)
    feature_df.set_index("subject_id", drop=True, inplace=True)

    num_columns = feature_df.shape[1]
    features = torch.empty((len(voc.word2idx), num_columns), dtype=torch.float32)

    mapping_df = data_pd[["SUBJECT_ID", "HADM_ID"]]

    for occurrence_id, node_id in voc.word2idx.items():

        # extract raw occurrence id (drop prefix, cast)
        occurrence_id = int(occurrence_id[2:])

        # remap from visit id to subject id
        if co_occurrence == "visit":
            occurrence_id = mapping_df[mapping_df["HADM_ID"] == occurrence_id]["SUBJECT_ID"].values[
                0
            ]

        # extract features
        features[node_id] = torch.tensor(feature_df.loc[occurrence_id], dtype=torch.float32)

    # assemble and return features
    return nn.Parameter(features, requires_grad=False)


# ===============================================
#
# Co-Occurrence Links
#
# ===============================================
@dataclass(frozen=True)
class CoLinkConfig:
    """
    Class to hold configuration for co occurrence links

    Attributes
    ----------
    link_type: str
        {patient, visit}
    edge_weights: bool
        use edge weights or treat each edge with same weight
    normalize_weights: bool
        normalize such that incoming weights accumulate to 1.0
    alpha_intra: float
        hyperparameter to tune the contribution to the
        final features of intra node type messages e.g. disease
        to disease
    alpha_inter: float
        same as `alpha_intra` but between diff. types of
        nodes e.g. disease to prescription
    """

    link_type: str = "patient"
    edge_weights: bool = False
    normalize_weights: bool = True

    alpha_intra: float = 1.0
    alpha_inter: float = 1.0


def build_co_occurrence_links(
    data_pd: pd.DataFrame,
    disease_vocabulary: Vocabulary,
    prescription_vocabulary: Vocabulary,
    co_occurrence_vocabulary: Vocabulary,
    co_occurrence_type: str = "patient",
    edge_weights: bool = True,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Returns links between disease/prescription codes
    based on co occurrence in the provided dataset

    Parameter
    ---------
    data_pd: pd.DataFrame
        the dataset
    disease_vocabulary: Vocabulary
        holds the node ids for ICD9 codes
    prescription_vocabulary: Vocabulary
        holds the node ids for ATC4 codes
    co_occurrence_vocabulary: Vocabulary
        with the co-occurrence nodes (relevant if subsampling is used)
    co_occurrence_type: str
        {patient, visit}
    edge_weights: bool
        compute edge weights based on frequency
    normalize: bool
        whether to normalize the edge weights

    Return
    ------
    Tuple of tensors containing the edges
    """

    logging.info(f"[GRAPH] building co links for {co_occurrence_type}")
    node_data = build_co_occurrence_nodes_df(data_pd, co_occurrence_type)

    disease_edge_dict: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    prescr_edge_dict: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    cross_dp_edge_dict: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    cross_pd_edge_dict: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)

    disease_normalizer: Dict[int, int] = defaultdict(lambda: 0)
    prescription_normalizer: Dict[int, int] = defaultdict(lambda: 0)
    disease_cross_normalizer: Dict[int, int] = defaultdict(lambda: 0)
    prescription_cross_normalizer: Dict[int, int] = defaultdict(lambda: 0)

    for _, row in node_data.iterrows():

        # can be patient or visit of a patient depending on mode
        patient_id = row["ID"]

        # skip if not in vocabulary
        if patient_id not in co_occurrence_vocabulary.word2idx.keys():
            continue

        # get code sets
        disease_codes = [disease_vocabulary.word2idx[d] for d in row["ICD9_CODE"]]
        prescription_codes = [prescription_vocabulary.word2idx[p] for p in row["ATC4"]]

        # build disease-disease edges
        for d1, d2 in product(disease_codes, disease_codes):
            if d1 == d2:
                continue

            disease_edge_dict[(d1, d2)] += 1
            disease_normalizer[d2] += 1  # normalize incoming count

        # build prescr-prescr edges
        for p1, p2 in product(prescription_codes, prescription_codes):
            if p1 == p2:
                continue

            prescr_edge_dict[(p1, p2)] += 1
            prescription_normalizer[p2] += 1

        # build cross edges
        for d, p in product(disease_codes, prescription_codes):

            cross_dp_edge_dict[(d, p)] += 1
            prescription_cross_normalizer[p] += 1

            cross_pd_edge_dict[(p, d)] += 1
            disease_cross_normalizer[d] += 1

    # no edge weights, just return edges
    data = {
        "d2d": transpose_edges(list(disease_edge_dict.keys()), False),
        "p2p": transpose_edges(list(prescr_edge_dict.keys()), False),
        "d2p": transpose_edges(list(cross_dp_edge_dict.keys()), False),
        "p2d": transpose_edges(list(cross_pd_edge_dict.keys()), False),
    }

    for k, v in data.items():
        logging.info(f"[GRAPH] co links {k}: {len(v[0])}")

    # compute get edge weights
    if edge_weights:

        def compute_weights(transposed_edge_dict, edge_dict, normalizer_dict):
            weights = torch.zeros(len(edge_dict), dtype=torch.float32)
            for i in range(len(transposed_edge_dict[0])):
                normalizer = normalizer_dict[transposed_edge_dict[1][i]] if normalize else 1
                weights[i] = (
                    edge_dict[(transposed_edge_dict[0][i], transposed_edge_dict[1][i])] / normalizer
                )
            return weights

        data["d2d_weights"] = compute_weights(data["d2d"], disease_edge_dict, disease_normalizer)
        data["p2p_weights"] = compute_weights(
            data["p2p"], prescr_edge_dict, prescription_normalizer
        )
        data["d2p_weights"] = compute_weights(
            data["d2p"], cross_dp_edge_dict, prescription_cross_normalizer
        )
        data["p2d_weights"] = compute_weights(
            data["p2d"], cross_pd_edge_dict, disease_cross_normalizer
        )

        logging.info(f"[GRAPH] co link edge weights computed, normalized: {normalize}")

    return data
