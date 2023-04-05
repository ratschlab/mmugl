# ===============================================
#
# UMLS Database
#
# ===============================================
import copy
import logging
import pickle
from collections import defaultdict
from itertools import product, repeat
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from kg.data.graph import CodeTokenizer, build_co_occurrence_nodes_df, transpose_edges
from kg.data.vocabulary import SimpleTokenizer, Vocabulary

# Globals
MRRCONSO_USED_COLUMNS = [
    "CUI",
    "LAT",
    "TTY",
    "CODE",
    "STR",
    "SAB",
    "ISPREF",
    "SUPPRESS",
]
MRRCONSO_COLUMNS = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]

MRREL_USED_COLUMNS = ["CUI1", "CUI2", "SAB", "REL", "RELA", "SUPPRESS"]
MRREL_COLUMNS = [
    "CUI1",
    "AUI1",
    "STYPE1",
    "REL",
    "CUI2",
    "AUI2",
    "STYPE2",
    "RELA",
    "RUI",
    "SRUI",
    "SAB",
    "SL",
    "RG",
    "DIR",
    "SUPPRESS",
    "CVF",
]


# Full list at:
# https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html
UMLS_KEEP_VOCABULARIES = [
    "ICD9CM",
    "ATC",  # base hierarchies
    "BI",  # Beth Israel Clinical Problem List
    "NCI_CDISC",  # CDISC Terminology Standard
    "CCC",  # Clinical Care Classification
    "CCS",  # Clinical Classification Software
    "RAM",  # Clinical Concepts by R A Miller
    "CCPSS",  # Clinical Problem Statement,
    "CPT",  # Current Procedural Terminology
    "CSP",  # CRISP Thesaurus
    "DRUGBANK",  # DrugBank,
    "DXP",  # DXplain
    "FMA",  # Foundational Model of Anatomy
    "ICNP",  # International Classification for Nursing Practice
    "ICF",  # International Classification of Functioning, Disability and Health
    "MeSH",  # MeSH
    "NIC",  # Nursing Intervention Clf
    "NOC",  # Nursing Outcome Clf,
    "PCDS",  # Patient Care Dataset
    "RXNORM",  # RxNorm
    "SNOMEDCT_US",  # SNOMED CT US Edition
    "WHO",  # WHOART adverse drug interactions
]


def load_mrconso(
    path: str,
    used_columns: List[str] = MRRCONSO_USED_COLUMNS,
    memory_map: bool = True,
    drop_suppressed: bool = True,
) -> pd.DataFrame:
    """
    Load UMLS MRCONSO.RRF

    Parameter
    ---------
    path: path to MRCONSO.RRF
    used_columns: columns to extract
    memory_map: use mem mapped I/O
    drop_suppressed: drop according to SUPPRESS column

    Returns
    -------
    DataFrame containing the specified columns of MRCONSO
    """

    conso_df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=MRRCONSO_COLUMNS,
        dtype=str,
        engine="c",
        index_col=False,
        memory_map=memory_map,
        usecols=used_columns,
    )

    if drop_suppressed:
        conso_df = conso_df[conso_df["SUPPRESS"] == "N"]

    logging.info(f"[UMLS] loaded MRCONSO: {conso_df.shape}")
    return conso_df


def load_mrrel(
    path: str,
    used_columns: List[str] = MRREL_USED_COLUMNS,
    memory_map: bool = True,
    drop_suppressed: bool = True,
) -> pd.DataFrame:
    """
    Load UMLS MRREL.RRF

    Parameter
    ---------
    path: path to MRREL.RRF
    used_columns: columns to extract
    memory_map: use mem mapped I/O
    drop_suppressed: drop according to SUPPRESS column

    Return
    ------
    DataFrame containing the specified columns of MRREL
    """

    rel_df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=MRREL_COLUMNS,
        dtype=str,
        engine="c",
        index_col=False,
        memory_map=memory_map,
        usecols=used_columns,
    )

    if drop_suppressed:
        rel_df = rel_df[rel_df["SUPPRESS"] == "N"]

    logging.info(f"[UMLS] loaded MRREL: {rel_df.shape}")
    return rel_df


def exclude_vocabularies(df: pd.DataFrame, vocabularies: List[str]) -> pd.DataFrame:
    """
    Exclude some vocabularies from a UMLS table

    Parameter
    ---------
    df: DataFrame of a UMLS table
    vocabularies: list of vocabs to exclude

    Return
    ------
    The pruned DataFrame
    """

    logging.info(f"[UMLS] Exclude vocabularies: {vocabularies}")
    initial_len = len(df)
    for vocabulary in vocabularies:
        df = df[df["SAB"] != vocabulary]

    final_len = len(df)
    logging.info(f"[UMLS]\tdropped {initial_len - final_len} entries")

    return df


def keep_vocabularies(df: pd.DataFrame, vocabularies: List[str]) -> pd.DataFrame:
    """
    Drop all but the vocabularies in the provided list
    Returns a copy of the reduced DataFrame

    Parameter
    ---------
    df: DataFrame of a UMLS table
    vocabularies: list of vocabs to keep

    Return
    ------
    The pruned DataFrame
    """

    logging.info(f"[UMLS] Keep vocabularies: {vocabularies}")
    vocabularies = set(vocabularies)  # type: ignore

    initial_len = len(df)
    indicator = df["SAB"].apply(lambda x: x in vocabularies)
    df = df[indicator]

    final_len = len(df)
    logging.info(f"[UMLS]\tdropped {initial_len - final_len} entries")

    return df.copy()


def match_code_set_to_cuis(
    conso_df: pd.DataFrame,
    codes: Union[Sequence[str], Set[str]],
    umls_vocabulary: str = "ICD9CM",
) -> Dict[str, Dict[str, str]]:
    """
    Match a set of codes to their concept CUIs in UMLS

    Parameter
    ---------
    conso_df: MRCONSO as DataFrame
    codes: ICD oder ATC codes to match
    umls_vocabulary: {'ICD9CM', 'ATC'}

    Return
    ------
    A mapping from codes to CUIs with STR information
    """

    logging.info(f"[UMLS] Matching codes to {umls_vocabulary}")

    # keep only relevant vocabulary, preferred and hierarchical terms
    conso_codes_df = conso_df[conso_df["SAB"] == umls_vocabulary]
    conso_codes_df = conso_codes_df[
        (conso_codes_df["TTY"] == "PT") | (conso_codes_df["TTY"] == "HT")
    ]

    conso_codes_df["CODE_stripped"] = conso_codes_df["CODE"].apply(lambda x: x.replace(".", ""))
    conso_codes_df["CODE_stripped_parents"] = conso_codes_df["CODE"].apply(
        lambda x: x[: x.rfind(".")]
    )

    umls_codes_set = set(conso_codes_df["CODE_stripped"].values)
    umls_codes_set_parents = set(conso_codes_df["CODE_stripped_parents"].values)
    mapping = {}

    def find_and_add_match(code: str, map_code=None):

        # dealing with higher level ICD9 term
        if "-" in code:
            matches = conso_codes_df[conso_codes_df["CODE_stripped_parents"] == code]
        else:
            matches = conso_codes_df[conso_codes_df["CODE_stripped"] == code]

        if map_code is None:
            map_code = code
        if len(matches) > 1:
            for _, row in matches.iterrows():
                if row["CODE"][2] != ".":
                    mapping[map_code] = {"cui": row["CUI"], "str": row["STR"]}
        else:
            mapping[map_code] = {
                "cui": matches.iloc[0]["CUI"],
                "str": matches.iloc[0]["STR"],
            }

    total_codes = len(codes)
    matched_codes = 0
    unmatched_codes = []

    # find matches
    for i, code in enumerate(tqdm(codes)):
        if code in umls_codes_set:
            matched_codes += 1
            find_and_add_match(code)

        elif "-" in code and code in umls_codes_set_parents:
            matched_codes += 1
            find_and_add_match(code)

        else:
            unmatched_codes.append(code)

    logging.info(f"[UMLS]\tMatched {matched_codes}/{total_codes}")
    logging.info(f"[UMLS]\tUnmatched codes: {unmatched_codes}")

    # Perform second pass trying to find higher level match (for ICD)
    for code, code_orig in map(lambda x: (x[:-1], x), unmatched_codes):
        if code in umls_codes_set:
            matched_codes += 1
            find_and_add_match(code, map_code=code_orig)
        else:
            logging.info(f"[UMLS]\t2nd pass unmatched: {code}")

    logging.info(f"[UMLS]\tFinal matched {matched_codes}/{total_codes}")
    logging.info(f"[UMLS]\tMapping size: {len(mapping.keys())}")

    return mapping


def compute_partition_complete_edges(data: Tuple) -> Sequence[Tuple[str, ...]]:
    """
    Extracts edges contained in `df` if
    both ends are present in `nodes`

    Parameter
    ---------
    data: tuple
        data.worker_id: parallel worker id
        data.nodes: set of nodes of the current graph
        data.df: chunk of MRREL with edges
        data.add_sab: add SAB as edge attribute

    Return
    ------
    List of matched edge tuples
    """

    worker_id, nodes, df, add_sab = data
    edges: List[Tuple[str, ...]] = []

    iterator = df.iterrows()
    if worker_id == 0:
        iterator = tqdm(iterator, total=len(df))

    for _, row in iterator:

        n1 = row["CUI1"]
        n2 = row["CUI2"]

        if n1 in nodes and n2 in nodes:

            if add_sab:
                edges.append((n1, n2, row["RELA"], row["SAB"]))
            else:
                edges.append((n1, n2, row["RELA"]))

    return edges


def compute_partition_1hop(data: Tuple) -> Sequence[Tuple[str, ...]]:
    """
    Extracts edges contained in `df` if
    both one of the two ends are present in `nodes`

    Parameter
    ---------
    data: tuple
        data.worker_id: parallel worker id
        data.nodes: set of nodes of the current graph
        data.df: chunk of MRREL with edges
        data.add_sab: add SAB as edge attribute

    Return
    ------
    List of matched edge tuples
    """

    worker_id, nodes, df, add_sab = data
    edges = []

    iterator = df.iterrows()
    if worker_id == 0:
        iterator = tqdm(iterator, total=len(df))

    for _, row in iterator:

        n1 = row["CUI1"]
        n2 = row["CUI2"]

        if n1 in nodes or n2 in nodes:
            if add_sab:
                edges.append((n1, n2, row["RELA"], row["SAB"]))  # type: ignore
            else:
                edges.append((n1, n2, row["RELA"]))  # type: ignore

    return edges


def add_edges_to_graph(
    graph: nx.Graph,
    rel_df: pd.DataFrame,
    conso_df: pd.DataFrame,
    extraction: str = "complete",
    workers: int = 4,
    add_sab: bool = False,
) -> nx.Graph:
    """
    Adds edges (undirected) to the graph

    Parameter
    ---------
    graph: -
    rel_df: MRREL as DataFrame
    conso_df: MRCONSO as DataFrame
    extraction: {'complete', '1hop'}
    workers: Parallel worker processes to use
    add_sab: add SAB as edge attribute
    """

    conso_index_df = conso_df.set_index(keys="CUI")

    nodes = set(graph.nodes)
    df_split = np.array_split(rel_df, workers)
    pool = Pool(workers)

    if extraction == "complete":
        compute_function = compute_partition_complete_edges
    elif extraction == "1hop":
        compute_function = compute_partition_1hop
    else:
        logging.error(f"[ERROR] unsupported extraction method: {extraction}")
        exit()

    logging.info(
        f"[UMLS] Extract `{extraction}` edge matches, parse {len(rel_df)} entries on {workers} workers"
    )
    chunk_results = pool.map(
        compute_function,
        zip(range(workers), repeat(nodes, workers), df_split, repeat(add_sab, workers)),
    )

    matched_edges: List[Tuple[Any, ...]] = []
    for chunk in chunk_results:
        matched_edges += chunk

    pool.close()
    pool.join()

    logging.info(f"[UMLS]\tMatched {len(matched_edges)} edges")

    skipped_edges = 0
    if add_sab:
        for n1, n2, rela, sab in tqdm(matched_edges):
            try:
                conso_index_df.loc[n1]
                conso_index_df.loc[n2]
                graph.add_edge(n1, n2, rela=rela, sab=sab)
            except:
                skipped_edges += 1
                continue
    else:
        for n1, n2, rela in tqdm(matched_edges):
            try:
                conso_index_df.loc[n1]
                conso_index_df.loc[n2]
                graph.add_edge(n1, n2, rela=rela)
            except:
                skipped_edges += 1
                continue

    logging.info(f"[UMLS] skipped edges due to CUI Index: {skipped_edges}")
    logging.info(f"[UMLS]\tUpdated {graph}")
    return graph


def add_attributes(
    graph: nx.Graph,
    conso_df: pd.DataFrame,
    conso_field: str = "str",
    label_field: str = "str",
    drop_not_found: bool = False,
    sort_str_len: bool = True,
) -> nx.Graph:
    """
    Extract attributes for each node in Graph from CONSO.RRF

    Parameter
    ---------
    graph: -
    conso_df: MRCONSO as DataFrame
    conso_field: MRCONSO field name
    label_field: label name in graph
    drop_not_found: drop nodes not found in MRCONSO
    sort_str_len: bool
        sort by string length

    Return
    ------
    Graph with extracted attributes from MRCONSO
    """

    conso_df = conso_df.set_index(keys="CUI")
    updated_nodes = 0
    dropped_nodes = 0
    none_nodes = 0

    logging.info(f"[UMLS] Extract {conso_field} into {label_field}")
    for node in tqdm(copy.deepcopy(graph.nodes)):

        attr = graph.nodes[node]
        if label_field not in attr:

            # try to get node
            # drop if not present in given MRCONSO
            try:
                data = conso_df.loc[node]
            except KeyError:
                if drop_not_found:
                    graph.remove_node(node)
                    dropped_nodes += 1
                    continue
                else:
                    data = None

            if data is None:
                if label_field == "str":
                    attr[label_field] = "none"
                else:
                    attr[label_field] = None
                none_nodes += 1
                nx.set_node_attributes(graph, {node: attr})
                continue

            # add attribute to graph
            if len(data.shape) > 1:
                try:
                    if label_field == "str":
                        data = data[(data["SUPPRESS"] == "N")].copy()
                        if sort_str_len:
                            data['STR_LEN'] = data[conso_field].str.len()
                            data = data.sort_values('STR_LEN', ascending=False)
                        data = data.iloc[0]
                    else:
                        data = data[(data["SUPPRESS"] == "N")].iloc[0]
                except IndexError:
                    graph.remove_node(node)
                    dropped_nodes += 1
                    continue

            data = data[conso_field]
            attr[label_field] = data
            nx.set_node_attributes(graph, {node: attr})
            updated_nodes += 1

    logging.info(f"[UMLS]\tupdated {updated_nodes} nodes")
    logging.info(f"[UMLS]\tdropped nodes {dropped_nodes}")
    logging.warning(f"[UMLS]\tNone nodes {none_nodes}")
    return graph


def add_sapbert_embeddings(
        graph: nx.Graph,
        workers: int = -1,
        batch_size: int = 128,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        avg_method: str = "cls") -> nx.Graph:
    """
    Retrieve a SapBert: https://github.com/cambridgeltl/sapbert
    embedding for each node (requires STR/str field in the graph data)
    and add as new node attribute under `sapbert`

    Parameter
    ---------
    graph: -
    workers: worker threads for the model
    batch_size: -
    model_name: str
        directory path to load pretrained LM from
    avg_method: str
        token averaging method {cls, pooling}

    Return
    ------
    Graph with annotated SapBert embeddings
    """

    # get current node data
    node_list = list(graph.nodes(data=True))

    # get str
    cui_list = list(map(lambda x: x[0], node_list))
    str_list = list(map(lambda x: x[1]["str"], node_list))

    if workers > 0:
        torch.set_num_threads(workers)

    # load model and tokenizer
    logging.info(f"[NODE EMBEDDINGS] Load SapBERT model, inference on {workers} threads")
    logging.info(f"[NODE EMBEDDINGS] load model from: {model_name}")
    logging.info(f"[NODE EMBEDDINGS] averaging method: {avg_method}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    bs = batch_size
    all_reps = []
    for i in tqdm(np.arange(0, len(str_list), bs)):

        toks = tokenizer.batch_encode_plus(
            str_list[i : i + bs],
            padding="max_length",
            max_length=64, # 32,
            truncation=True,
            return_tensors="pt",
        )

        output = model(**toks)  # pass tokens through model

        if avg_method == "cls":
            representations = output[0][:, 0, :]  # get CLS representation

        else: # average pool all unmasked tokens
            token_embeddings = output[0]
            attention_mask = toks['attention_mask'].cpu()
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            representations = sum_embeddings / sum_mask

        all_reps.append(representations.cpu().detach().numpy())

    # collect embeddings
    all_reps_emb = np.concatenate(all_reps, axis=0)

    sapbert_attr = {}
    for i, cui in enumerate(cui_list):
        sapbert_attr[cui] = {"sapbert": all_reps_emb[i, :]}

    nx.set_node_attributes(graph, sapbert_attr)

    logging.info(f"Added embeddings to {len(node_list)} nodes")
    return graph


def build_umls_graph_from_networkx(
    graph: nx.Graph, vocabulary: Vocabulary = None
) -> Tuple[Vocabulary, List]:
    """
    Builds a vocabulary (mapping from node CUIs to integer indeces)
    and an edge tuple matrix from a networkx graph

    Parameter
    ---------
    graph: the UMLS networkx graph
    vocabulary: Vocabulary
        vocabulary to use for node indexing

    Return
    ------
    cui_vocab: Vocabulary
        the vocabulary, if vocabulary Parameter is provided
        the same is returned
    cui_edges: List
        the graph edges as edge tuple matrix
    """

    # build node vocabulary, assign interger ids
    if vocabulary is None:
        cui_vocab = Vocabulary()
        cui_vocab.add_sentence(graph.nodes)
        logging.info(f"[UMLS] loaded CUI vocabulary: {len(cui_vocab.idx2word)}")
    else:
        cui_vocab = vocabulary
        logging.info(f"[UMLS] use provided CUI vocabulary: {len(cui_vocab.idx2word)}")

    # build edges
    edge_idx = []
    for n1, n2 in graph.edges:

        i1 = cui_vocab.word2idx[n1]
        i2 = cui_vocab.word2idx[n2]

        edge_idx.extend([(i1, i2), (i2, i1)])

    cui_edges = transpose_edges(edge_idx)
    logging.info(f"[UMLS] loaded {len(cui_edges[0])} edges")

    return cui_vocab, cui_edges


def build_umls_co_occurrence_edges(
    data_pd: pd.DataFrame,
    co_occurrence_vocabulary: Vocabulary,
    cui_vocabulary: Vocabulary,
    icd9_to_cui_map: Dict,
    atc_to_cui_map: Dict,
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
    cui_vocabulary: `Vocabulary` of UMLS CUI nodes
    icd9_to_cui_map: -
    atc_to_cui_map: -
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
    cui2node_edge_idx = []
    node2cui_edge_idx = []

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
        node2cui_edge_idx.extend(
            [(node, cui_vocabulary.word2idx[icd9_to_cui_map[d]["cui"]]) for d in disease_codes]
        )
        cui2node_edge_idx.extend(
            [(cui_vocabulary.word2idx[icd9_to_cui_map[d]["cui"]], node) for d in disease_codes]
        )

        # build edges to prescription nodes, undirected
        prescription_codes = row["ATC4"]
        node2cui_edge_idx.extend(
            [(node, cui_vocabulary.word2idx[atc_to_cui_map[p]["cui"]]) for p in prescription_codes]
        )
        cui2node_edge_idx.extend(
            [(cui_vocabulary.word2idx[atc_to_cui_map[p]["cui"]], node) for p in prescription_codes]
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

    logging.info(f"Extracted {len(node2cui_edge_idx)} node<->disease co-occurrence edges")

    node2cui_edge_idx = transpose_edges(node2cui_edge_idx, remove_duplicates=False)  # type: ignore
    cui2node_edge_idx = transpose_edges(cui2node_edge_idx, remove_duplicates=False)  # type: ignore

    data_dict = {
        "node2cui": node2cui_edge_idx,
        "cui2node": cui2node_edge_idx,
    }

    if co_occurrence_loss > 0.0:
        data_dict["d_targets"] = co_occurrence_target_d  # type: ignore
        data_dict["p_targets"] = co_occurrence_target_p  # type: ignore

        d_hash = co_occurrence_target_d.sum()
        p_hash = co_occurrence_target_p.sum()
        logging.info(f"Extracted co-occurrence targets d: {co_occurrence_target_d.shape}:{d_hash}")
        logging.info(f"Extracted co-occurrence targets p: {co_occurrence_target_p.shape}:{p_hash}")

    return data_dict


def build_sapbert_node_matrix(
    graph: nx.Graph,
    voc: Vocabulary,
    special_tokens: Set[str] = {"[PAD]", "[CLS]", "[MASK]"},
) -> torch.Tensor:
    """
    Computes the feature matrix for the CUI nodes
    in `voc` using the node attributes of `graph`.`sapbert`

    Parameter
    ---------
    graph: -
    voc: CUI vocabulary

    Return
    ------
    feature matrix
    """

    graph_data = graph.nodes(data=True)
    node2sapbert = {}
    for cui, attr in graph_data:
        node2sapbert[cui] = attr["sapbert"]

    # get dimensions
    sapbert_dim = len(list(graph_data)[0][1]["sapbert"])
    vocab_size = len(voc.idx2word)
    feature_matrix = torch.zeros((vocab_size, sapbert_dim), dtype=torch.float32)

    for i, cui in voc.idx2word.items():
        if cui in special_tokens:
            continue
        feature_matrix[i, :] = torch.tensor(node2sapbert[cui])

    logging.info(f"[UMLS] Built SapBERT feature matrix: {feature_matrix.shape}")
    return torch.nn.Parameter(feature_matrix, requires_grad=False)


def get_cuis_from_text_concept_raw_data(
    filepath: str, threshold: float = 0.0, categories: Iterable[str] = None
) -> Set[str]:
    """
    Get a set of UMLS-CUIs contained in a result of
    concept extraction from MIMIC-III data

    Parameter
    ---------
    filepath: str
        path to extracted concept file
    threshold: float
        threshold for similarity metric to include
    categories: Iterable[str]
        set of categories to consider, if not provide
        will use all
    """

    logging.info(f"[UMLS] loading concept-text file: {filepath.split('/')[-1]}")
    with open(filepath, "rb") as file:
        data = pickle.load(file)

    cuis: Set[str] = set()
    cats = None if categories is None else set(categories)
    logging.info(f"[UMLS] categories: {categories}")

    for patient in tqdm(data.values()):
        for visit in patient.values():
            for category, category_documents in visit.items():

                # if a set of categories is provided
                # we skip the category if not in list
                if cats is not None and category not in cats:
                    continue

                for document in category_documents:
                    cuis = cuis.union(
                        map(
                            lambda match: match[0],
                            filter(lambda match: match[1] >= threshold, document),
                        )
                    )

    logging.info(f"[UMLS] extracted {len(cuis)} different CUIs")
    return cuis


def load_umls_graph(path: str, get_concept_tokenizer: bool = False) -> Dict[str, Any]:
    """
    Load a networkx graph from path

    Parameter
    ---------
    path: str
        path to load graph from using pickle
    get_concept_tokenizer: bool
        also return a tokenizer over the nodes
    """

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    if not get_concept_tokenizer:
        return {
            "graph": data["graph"],
            "grow_hops": data["grow_hops"],
            "icd9_to_cui_map": data["icd9_to_cui_map"],
            "atc_to_cui_map": data["atc_to_cui_map"],
        }

    tokenizer = SimpleTokenizer(codes=data["graph"].nodes)
    return {
        "graph": data["graph"],
        "grow_hops": data["grow_hops"],
        "icd9_to_cui_map": data["icd9_to_cui_map"],
        "atc_to_cui_map": data["atc_to_cui_map"],
        "tokenizer": tokenizer,
    }


# ===============================================
#
# Co-Occurrence Links
#
# ===============================================
def build_umls_co_occurrence_links(
    data_pd: pd.DataFrame,
    cui_vocabulary: Vocabulary,
    icd9_to_cui_map: Dict,
    atc_to_cui_map: Dict,
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
    cui_vocabulary: Vocabulary
        holds the node ids for the CUI (UMLs) nodes
    icd9_to_cui_map: Dict
        maps ICD9 codes to UMLS CUIs
    atc_to_cui_map: Dict
        maps ATC codes to UMLS CUIs
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
        disease_codes = [
            cui_vocabulary.word2idx[icd9_to_cui_map[d]["cui"]] for d in row["ICD9_CODE"]
        ]
        prescription_codes = [
            cui_vocabulary.word2idx[atc_to_cui_map[p]["cui"]] for p in row["ATC4"]
        ]

        # build disease-disease edges
        for d1, d2 in product(disease_codes, disease_codes):
            if d1 == d2:
                continue

            disease_edge_dict[(d1, d2)] += 1
            disease_normalizer[d2] += 1  # normalize incoming count

            # disease_edge_dict[(d2, d1)] += 1
            # disease_normalizer[d1] += 1 # normalize incoming count

        # build prescr-prescr edges
        for p1, p2 in product(prescription_codes, prescription_codes):
            if p1 == p2:
                continue

            prescr_edge_dict[(p1, p2)] += 1
            prescription_normalizer[p2] += 1

            # prescr_edge_dict[(p2, p1)] += 1
            # prescription_normalizer[p1] += 1

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
