#!/usr/bin/env python3
# ===============================================
#
# Extract Medical Concepts from Clinical Reports
# in MIMIC-III with QuickUMLS
#
# ===============================================
from fileinput import filename
from json import load
import logging
import coloredlogs
import argparse
import pickle
import time

import os
import subprocess
from os import path
from typing import Dict, Any
from random import choice, sample
from multiprocessing import Pool
from itertools import repeat

# Tools
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from spacy.tokens import Span

# QuickUMLS
from quickumls import QuickUMLS
# from quickumls.constants import ACCEPTED_SEMTYPES

# Custom KG
from kg.data.umls import exclude_vocabularies, load_mrconso, match_code_set_to_cuis
from kg.data.processing import read_set_codes, load_noteevents_table
from kg.data.graph import build_atc_tree, build_icd9_tree
from kg.utils.constants import QUICKUMLS_ACCEPTED_SEMTYPES
from kg.data.quickumls import QUSettings, map_quickumls_from_text, map_quickumls_from_text_spacy
import kg.data.quickumls
from kg.data.eicu import process_note_table_eICU

# Globals
LOGGING_LEVEL = "INFO"


def parse_arguments() -> argparse.Namespace:

    # get parser
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("-o", "--output", required=True, type=str,
        help="Output folder")
    parser.add_argument("-n", "--name", default="base", type=str,
        help="Name, will be part of the output filename")
    parser.add_argument("-w", "--workers", default=1, type=int,
        help="Number of parallel worker processes to use")
    parser.add_argument("--test", default=-1, type=int,
        help="Slice of data for testing")
    parser.add_argument("--use_scratch", default=None, type=str,
        help="Use local scratch for QuickUMLS files")

    parser_data = parser.add_argument_group("Data")
    parser_data.add_argument("--mimic_path", required=True, type=str,
        help="Path to MIMIC-III data")
    parser_data.add_argument("--eicu_path", default=None, type=str,
        help="Path to eICU hdf5 files")

    # Grah related arguments
    parser_quick = parser.add_argument_group("QuickUMLS")
    parser_quick.add_argument("--quickumls_path", required=True, type=str,
        help="Path to QuickUMLS files")
    parser_quick.add_argument("--window", default=6, type=int,
        help="QuickUMLS window size")
    parser_quick.add_argument("-t", "--threshold", default=0.9, type=float,
        help="QuickUMLS threshold value")
    parser_quick.add_argument("-s", "--similarity", default='jaccard', type=str,
        help="QuickUMLS similarity metric: {jaccard, cosine, dice, overlap}")

    args = parser.parse_args()
    return args




def main():
    """Graph Extraciton Script procedure"""
    t0_full = time.time()

    # Parse CMD arguments
    args = parse_arguments()
    processing_eICU_dataset = args.eicu_path is not None

    # set QU path
    quickumls_path = args.quickumls_path

    # configure local scratch
    if args.use_scratch is not None:

        tmpdir = args.use_scratch
        logging.info(f"Using local scratch at {tmpdir}")

        process = subprocess.run(
            ['rsync', '-aq', args.quickumls_path,  tmpdir],
            stdout=subprocess.PIPE)

        if process.returncode != 0:
            logging.error(f"Rsync from: {args.quickumls_path} to local scratch failed: {process.returncode}")
            exit()
        else:
            logging.info(f"Copied QuickUMLS to local scratch")
            quickumls_path = tmpdir


    # get noteevents filename
    if not processing_eICU_dataset:
        logging.info(f"Processing MIMIC data")
        noteevents_file = path.join(args.mimic_path, 'NOTEEVENTS.csv')
        noteevents_df = load_noteevents_table(noteevents_file, drop_error=True)
    else:
        logging.info(f"Processing eICU data")
        noteevents_file = path.join(args.eicu_path, 'note.h5')
        patients_file = path.join(args.eicu_path, 'patient.h5')
        noteevents_df = process_note_table_eICU(noteevents_file, patients_file)
        
    if args.test > 0:
        noteevents_df = noteevents_df[:args.test]
        logging.warning(f"[TESTING] data shape: {noteevents_df.shape}")

    # split
    df_split = np.array_split(noteevents_df, args.workers)

    # worker pools
    pool = Pool(args.workers)

    # QuickUMLS settings
    logging.info(f"[QuickUMLS] {quickumls_path}")
    logging.info(f"[QuickUMLS] {args.similarity}@{args.threshold}, w: {args.window}")
    settings = QUSettings(quickumls_path, args.threshold,
        args.similarity, args.window)

    # map using parallel pool
    logging.info(f"[UMLS] parse {len(noteevents_df)} entries on {args.workers} workers")
    t0 = time.time()
    chunk_results = pool.map(map_quickumls_from_text_spacy,
        zip(range(args.workers), df_split, repeat(settings, args.workers)))
    t1 = time.time()
    logging.info(f"[TIME] took {((t1 - t0) / 60):.1f} minutes")

    logging.info(f"[UMLS] merging chunks")
    t0 = time.time()
    results = {}
    for chunk in chunk_results:
        for row in tqdm(chunk):

            patient_id = row[0]
            visit_id = row[1]
            category = row[2]
            cuis = row[3]

            if patient_id not in results:
                results[patient_id] = {}

            results_patient = results[patient_id]
            if visit_id not in results_patient:
                results_patient[visit_id] = {}

            results_visit = results_patient[visit_id]
            if category not in results_visit:
                results_visit[category] = []

            results_category = results_visit[category]
            results_category.append(cuis)

    t1 = time.time()
    logging.info(f"[TIME] took {((t1 - t0) / 60):.1f} minutes")

    filename = "noteevents"
    if processing_eICU_dataset:
        filename += "_eICU"
    filename += f"_{args.name}_{args.similarity}_{args.threshold}_w{args.window}.pkl"
    file_path = path.join(args.output, filename)
    logging.info(f"Store graph and maps to: {file_path}")
    with open(file_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    t1_full = time.time()
    logging.info(40*"=")
    logging.info(f"[TIME] took {((t1_full - t0_full) / 60):.1f} minutes")
    logging.info("Done")
    logging.info(40*"=")



if __name__ == "__main__":
    
    # set logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s | %(message)s')
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()