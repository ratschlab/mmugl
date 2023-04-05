#!/usr/bin/env python3
# ===============================================
#
# Create a knowledge graph from medical concepts
# extracted from clinical reports using UMLS
# relational information
#
# ===============================================
from fileinput import filename
import logging
import coloredlogs
import argparse
import pickle
from pathlib import Path

from os import path
from typing import Dict, Any
from random import choice, sample
from multiprocessing import Pool
from itertools import repeat
from kg.data.umls import exclude_vocabularies, load_mrconso, \
    match_code_set_to_cuis, get_cuis_from_text_concept_raw_data

# Tools
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# QuickUMLS
from quickumls import QuickUMLS
from quickumls.constants import ACCEPTED_SEMTYPES

# Custom KG
from kg.data.processing import read_set_codes
from kg.data.umls import *
from kg.data.graph import build_atc_tree, build_icd9_tree

# Globals
LOGGING_LEVEL = "INFO"
CATEGORIES = ['Discharge summary', 'Nursing', 'Radiology', 'eICU', 'Physician', 'Respiratory', 'General']


def parse_arguments() -> argparse.Namespace:

    # get parser
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("-v", "--verbose", action="store_true", help="More verbose output")
    parser.add_argument("-o", "--output", required=True, type=str,
        help="Output path/filename")
    parser.add_argument("-w", "--workers", default=1, type=int,
        help="Number of parallel worker processes to use")

    # Options to include text-concept mapped data
    parser.add_argument("--text_concepts", default=None, type=str,
        help="Path to extracted concepts from text notes data")
    parser.add_argument("--text_threshold", type=float, default=0.0)
    parser.add_argument("--use_categories", default=False, action='store_true',
        help="Whether to consider the script hard-coded category list; if none given, all will be considered")

    # Grah related arguments
    parser_graph = parser.add_argument_group("Graph")

    parser_graph.add_argument("--icd_codes", required=True, type=str,
        help="Path to .txt containing list of considered ICD codes")
    parser_graph.add_argument("--atc_codes", required=True, type=str,
        help="Path to .txt containing list of considered ATC codes")
    parser_graph.add_argument("--build_code_trees", action='store_true', default=False,
        help="Whether to build the ICD/ATC trees (with parent nodes) from the given code sets before matching to CUIs")

    parser_graph.add_argument("--umls_path", required=True, type=str,
        help="Path to UMLS directory, containing MRREF, MRCONSO,...")
    parser_graph.add_argument("--language", default='ENG', type=str,
        help="Language of entries to extract from UMLS database") 

    parser_graph.add_argument("--grow_hops", default=0, type=int,
        help="How many times to add 1-hop distant nodes to grow the graph")
    parser_graph.add_argument("--reduce_vocabularies", default=False, action='store_true',
        help="Only keep a curated list of vocabularies")
    parser_graph.add_argument("--add_sab", default=False, action='store_true',
        help="Add SAB entry to nodes")
    parser_graph.add_argument("--add_sapbert", default=False, action='store_true',
        help="Add SapBERT embeddings based on STR field")


    args = parser.parse_args()
    return args




def main():
    """Graph Extraciton Script procedure"""

    # Parse CMD arguments
    args = parse_arguments()

    # Read in target codes
    disease_codes = read_set_codes(args.icd_codes)
    if args.verbose: logging.info(f"Parsed {len(disease_codes)} diagnosis codes from provided file")
    prescription_codes = read_set_codes(args.atc_codes)
    if args.verbose: logging.info(f"Parsed {len(prescription_codes)} prescription codes from provided file")

    # build full code trees from base set
    if args.build_code_trees:
        disease_codes = list(build_icd9_tree(disease_codes)[1].word2idx.keys())
        logging.info(f"Built full ICD tree: {len(disease_codes)}")

        prescription_codes = list(build_atc_tree(prescription_codes)[1].word2idx.keys())
        logging.info(f"Built full ATC tree: {len(prescription_codes)}")

    # load conso, rref
    conso_df = load_mrconso(path.join(args.umls_path, "MRCONSO.RRF"),
        drop_suppressed=True)
    rel_df = load_mrrel(path.join(args.umls_path, "MRREL.RRF"),
        drop_suppressed=True)

    # get language
    conso_df = conso_df[conso_df['LAT'] == args.language]
    if args.verbose: logging.info(f"MRCONSO use language {args.language} entries: {conso_df.shape}")

    if args.reduce_vocabularies:
        logging.info("Reduce vocabulary with curated list")
        conso_df = keep_vocabularies(conso_df, UMLS_KEEP_VOCABULARIES)
        rel_df = keep_vocabularies(rel_df, UMLS_KEEP_VOCABULARIES)

    # drop some vocabularies
    # drop ICD10, as we use ICD9
    drop_vocabularies = [
        'DMDICD10', 'ICD10PCS', 'ICD10AE',
        'ICD10AM', 'ICD10AMAE', 'ICD10DUT']

    conso_df = exclude_vocabularies(conso_df, drop_vocabularies)
    rel_df = exclude_vocabularies(rel_df, drop_vocabularies)

    # get code to CUI mappings
    icd9_to_cui_map = match_code_set_to_cuis(
            conso_df, disease_codes, umls_vocabulary='ICD9CM')

    atc_to_cui_map = match_code_set_to_cuis(
            conso_df, prescription_codes, umls_vocabulary='ATC')
        

    # initialize graph
    G = nx.Graph()

    # add ICD nodes
    icd_cui_nodes = [(data['cui'], {'code': code, 'str': data['str']})
        for code, data in icd9_to_cui_map.items()]
    G.add_nodes_from(icd_cui_nodes)
    if args.verbose: logging.info(f"Add ICD nodes: {G}")

    # add ATC nodes
    atc_cui_nodes = [(data['cui'], {'code': code, 'str': data['str']})
        for code, data in atc_to_cui_map.items()]
    G.add_nodes_from(atc_cui_nodes)
    if args.verbose: logging.info(f"Add ATC nodes: {G}")

    # get cuis present in text files
    if args.text_concepts is not None:
        logging.info(f"Load CUIs from text notes")

        cats = CATEGORIES if args.use_categories else None
        text_cuis = get_cuis_from_text_concept_raw_data(
            args.text_concepts, args.text_threshold, cats)

        G.add_nodes_from(text_cuis)
        if args.verbose: logging.info(f"Add text nodes: {G}")


    # add edges to connect added ICD and ATC nodes
    # and potentially text nodes
    G = add_edges_to_graph(G, rel_df, conso_df, workers=args.workers,
        extraction='complete', add_sab=args.add_sab)

    # grow graph
    logging.info(f"Grow graph {args.grow_hops} times")
    for _ in range(args.grow_hops):
        G = add_edges_to_graph(G, rel_df, conso_df, workers=args.workers,
            extraction='1hop', add_sab=args.add_sab)

    # add attributes
    G = add_attributes(G, conso_df, conso_field='STR', label_field='str')

    if args.add_sab:
        G = add_attributes(G, conso_df, conso_field='SAB', label_field='sab')

    if args.add_sapbert:
        G = add_sapbert_embeddings(G, args.workers, 128)

    # save graph
    data = {
            'graph': G,
            'grow_hops': args.grow_hops,
            'umls_path': args.umls_path,
            'icd9_to_cui_map': icd9_to_cui_map,
            'atc_to_cui_map': atc_to_cui_map
        }

    logging.info(f"Store graph and maps to: {args.output}")

    # create directories along path
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(args.output, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    logging.info(40*"=")
    logging.info("Done")
    logging.info(40*"=")



if __name__ == "__main__":
    
    # set logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s | %(message)s')
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()