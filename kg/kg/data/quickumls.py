# ===============================================
#
# Utility around QuickUMLS
#
# ===============================================
import logging
from collections import namedtuple
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import spacy
from negspacy.negation import Negex
from quickumls import QuickUMLS
from spacy.language import Language
from spacy.tokens import Span, Token
from tqdm import tqdm

# KG
from kg.utils.constants import QUICKUMLS_ACCEPTED_SEMTYPES

QUSettings = namedtuple("QUSettings", ["quickumls_path", "threshold", "similarity", "window"])

default_qu_settigs = QUSettings(
    quickumls_path="please/provide/default/path/to/quickumls",
    threshold=0.9,
    similarity="jaccard",
    window=6,
)


def map_quickumls_from_text(data: Tuple) -> Sequence[Tuple[Any, ...]]:
    """
    For each row in data map the text notes to
    UMLS CUIs using QuickUMLS

    Parameter
    ---------
    data: namedtuple
        data.worker_id: int
            parallel worker id
        data.chunk_df: pd.DataFrame
            chunk of the noteevents dataframe
        data.settings: Dict[str, Any]
            quickumls settings
    """
    worker_id, chunk_df, settings = data

    matcher = QuickUMLS(
        settings.quickumls_path,
        overlapping_criteria="score",
        threshold=settings.threshold,
        similarity_name=settings.similarity,
        window=settings.window,
        accepted_semtypes=QUICKUMLS_ACCEPTED_SEMTYPES,
    )

    # get iterator over chunk
    iterator = chunk_df.iterrows()
    if worker_id == 0:
        iterator = tqdm(iterator, total=len(chunk_df))

    results = []
    for _, row in iterator:

        patient_id = row["SUBJECT_ID"]
        visit_id = row["HADM_ID"]
        category = row["CATEGORY"].strip()

        # get text and preprocess
        text = row["TEXT"]
        text = text.replace("\n", " ").replace("\r", "")  # remove linebreaks

        matches = matcher.match(text, best_match=True, ignore_syntax=False)

        document = set()
        cuis = set()
        for candidates in matches:

            # get best match
            match = candidates[0]

            # get attributes
            cui = match["cui"]
            similarity = match["similarity"]

            if cui not in cuis:
                cuis.add(cui)
                document.add((cui, similarity))

        results.append((patient_id, visit_id, category, document))

    return results


# ----------------------------------------------
#
# Spacy-based utilities
#
# ----------------------------------------------
class QUExtractor(object):
    """
    Spacy Compatible QuickUMLS Extractor Class

    Adapted from work by *Anonym*: *URL*
    specifically: *URL*
    """

    def __init__(self, nlp, qu_settings):

        # Create QuickUMLS Object
        self.quickumls = QuickUMLS(
            qu_settings.quickumls_path,
            overlapping_criteria="score",
            threshold=qu_settings.threshold,
            similarity_name=qu_settings.similarity,
            window=qu_settings.window,
            accepted_semtypes=QUICKUMLS_ACCEPTED_SEMTYPES,
        )

        # Save spacy pipeline
        self.nlp = nlp

        # let's extend this with some proprties that we want
        Span.set_extension("similarity", default=-1.0, force=True)
        Span.set_extension("semtypes", default=-1.0, force=True)
        Span.set_extension("cui", default="", force=True)
        Span.set_extension("ngram", default="", force=True)

    def __call__(self, doc):

        # Match UMLS vocabulary to document with ngrams and matches
        matches = self.quickumls._match(doc, best_match=True, ignore_syntax=False)

        # Convert QuickUMLS match objects into Spans

        # Integrate matches into document span for entity recognition
        for match in matches:

            # take first n_gram match
            ngram_match_dict = match[0]

            start_char_idx = int(ngram_match_dict["start"])
            end_char_idx = int(ngram_match_dict["end"])

            cui = ngram_match_dict["cui"]

            # char_span() creates a Span from these character indices
            # UMLS CUI should work well as the label here
            span = doc.char_span(start_char_idx, end_char_idx, label="UMLS", kb_id=str(cui))
            # add some custom metadata to the spans
            # if a span is already part of a different
            # recovered entity we skip it
            span._.similarity = ngram_match_dict["similarity"]
            span._.ngram = ngram_match_dict["ngram"]
            span._.semtypes = ngram_match_dict["semtypes"]
            span._.cui = cui
            try:
                if str(span) not in str(doc.ents):
                    doc.ents = list(doc.ents) + [span]
            except ValueError as e:
                logging.info(f"Could not add span: {e}")
                pass

        return doc


@Language.factory("QUExtractor")
def create_QUExtractor(nlp, name):
    return QUExtractor(nlp, name)


def map_quickumls_from_text_spacy(data: Tuple) -> Sequence[Tuple[Any, ...]]:
    """
    For each row in data map the text notes to
    UMLS CUIs using QuickUMLS

    Parameter
    ---------
    data: namedtuple
        data.worker_id: int
            parallel worker id
        data.chunk_df: pd.DataFrame
            chunk of the noteevents dataframe
        data.settings: Dict[str, Any]
            quickumls settings
    """
    worker_id, chunk_df, settings = data

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("QUExtractor", settings)
    nlp.add_pipe("negex", config={"ent_types": ["UMLS"]})

    # get iterator over chunk
    iterator = chunk_df.iterrows()
    if worker_id == 0:
        iterator = tqdm(iterator, total=len(chunk_df))

    results = []
    for _, row in iterator:

        patient_id = row["SUBJECT_ID"]
        visit_id = row["HADM_ID"]
        category = row["CATEGORY"].strip()

        # get text and preprocess
        text = row["TEXT"]
        text = text.replace("\n", " ").replace("\r", "")  # remove linebreaks
        matches = nlp(text)

        document = set()
        cuis = set()
        for candidates in matches.ents:

            # get attributes
            cui = candidates._.cui
            similarity = candidates._.similarity
            negation = -1 if candidates._.negex else 1

            if cui not in cuis and cui != "":
                cuis.add(cui)
                document.add((cui, similarity, negation))

        results.append((patient_id, visit_id, category, document))

    return results
