# ===============================================
#
# Torch Datasets for Training
#
# ===============================================
import logging
import pickle
import random
import time
from datetime import datetime
from os import path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kg.data.analysis import diagnosis_statistics
from kg.data.contants import HEART_FAILURE_PREFIX
from kg.data.eicu import load_patient_table_eICU
from kg.data.processing import (
    filter_by_visit_codes,
    filter_by_visit_range,
    filter_diagnosis_table,
    filter_patients_number_codes,
    filter_patients_number_codes_icd,
    get_unique_codes,
    group_diagnosis_table,
    group_table,
    load_and_preprocess_notes_concepts,
    map_ndc_atc4_codes,
    match_diagnosis_prescriptions_ids,
    merge_diagnosis_prescriptions,
    process_diagnosis_table,
    process_prescription_table,
    split_dataset,
)
from kg.data.vocabulary import (
    CodeTokenizer,
    DiagnosisTokenizer,
    SimpleTokenizer,
    Vocabulary,
)


# ===================================
#
# Utility
#
# ===================================
def random_word_masking(
    tokens: np.ndarray, vocabulary: Vocabulary, masking_probability: float = 0.15
) -> np.ndarray:
    """
    Randomly selects tokens with `masking_probability`. The selected tokens
    get masked 80%, randomly replaced 10% or left untouched 10% of the time.

    Parameters
    ----------
    tokens: numpy array of token ids
    vocabulary: the associated vocabulary used for tokenization
    masking_probability: probability for a token to be selected for masking

    Returns
    -------
    Numpy array of the masked sequence
    """
    mask_idx = vocabulary.word2idx["[MASK]"]
    mask = np.random.choice(
        [True, False],
        size=len(tokens),
        p=[masking_probability, 1.0 - masking_probability],
    )
    mask_token = np.random.choice([mask_idx, -1, -2], size=len(tokens), p=[0.8, 0.1, 0.1])

    for i, (m, choice) in enumerate(zip(mask, mask_token)):
        if m:
            if choice == mask_idx:
                tokens[i] = mask_idx
            elif choice == -1:
                tokens[i] = random.choice(list(vocabulary.idx2word.items()))[0]

    return tokens


# ===================================
#
# Datasets (General)
#
# ===================================
class DiagnosisDataset(Dataset):
    """
    Dataset for graph pretraining solely using diagnosis codes

    Attributes
    ----------
    diagnosis_pd: DataFrame holding the Datasets source
    tokenizer: the associated `DiagnosisTokenizer`
    xxx_token_idx: ids for special tokens
    random_masking_probability: probability for random masking
        ignored if 0.0
    max_sequence_length: output gets cut/padded to this length
    """

    def __init__(
        self,
        data_path: str,
        top_k_codes: int = 2048,
        patient_ids: Union[Set[int], Sequence[int]] = None,
        code_count_range: Tuple[float, float] = (2, np.inf),
        visit_range: Tuple[float, float] = None,
        max_sequence_length: int = 47,
        random_masking_probability: float = 0.0,
        diagnosis_codes: Union[Set[str], Sequence[str]] = None,
        tokenizer: DiagnosisTokenizer = None,
        code_shuffle: bool = True,
    ):
        """
        Constructor for `DiagnosisDataset`

        Parameters
        ----------
        data_path: path to source MIMIC-III `DIAGNOSES_ICD.csv`
        top_k_codes: only keep the top-k ICD9 codes, skipped if `None`
        patient_ids: list of patients to use in this dataset
        code_count_range: drop patients with a visit containing
            a code count outside of this range
        visit_range: only keep patients with number of visits within this range
        max_sequence_length: cut/pad output to this length
        random_masking_probability: probability for random masking,
        diagnosis_codes: set of disease codes used for dataset
        tokenizer: pass a `DiagnosisTokenizer`, will create one if None
        code_shuffle: shuffle visit codes for each load
        """
        super().__init__()

        self.diagnosis_pd = process_diagnosis_table(data_path)

        # only keep allowed patient records
        if patient_ids is not None:
            patient_ids = set(patient_ids)
            filter_lambda = lambda x: x["SUBJECT_ID"] in patient_ids
            self.diagnosis_pd = self.diagnosis_pd[self.diagnosis_pd.apply(filter_lambda, axis=1)]
            logging.info(f"Filtered patient ids, filtered shape: {self.diagnosis_pd.shape}")

        self.diagnosis_pd = filter_diagnosis_table(self.diagnosis_pd, num=top_k_codes)
        if diagnosis_codes is not None:
            self.diagnosis_pd = filter_by_visit_codes(self.diagnosis_pd, diagnosis_codes)

        self.diagnosis_pd = group_diagnosis_table(self.diagnosis_pd)
        self.diagnosis_pd = filter_patients_number_codes(
            self.diagnosis_pd, code_range=code_count_range
        )

        if visit_range is not None:
            self.diagnosis_pd = filter_by_visit_range(self.diagnosis_pd, visit_range=visit_range)

        if diagnosis_codes is None:
            logging.info("Compute set of codes to use in vocabulary from data")
            unique_codes = get_unique_codes(self.diagnosis_pd)
        else:
            logging.info(f"Use provided set of codes in vocabulary: {len(diagnosis_codes)}")
            unique_codes = diagnosis_codes

        if tokenizer is None:
            logging.info("Instatiating new tokenizer")
            self.tokenizer = DiagnosisTokenizer(unique_codes)
        else:
            logging.info("Use existing tokenizer")
            self.tokenizer = tokenizer

        self.pad_token_idx = self.tokenizer.vocabulary.word2idx["[PAD]"]
        self.mask_token_idx = self.tokenizer.vocabulary.word2idx["[MASK]"]
        self.cls_token_idx = self.tokenizer.vocabulary.word2idx["[CLS]"]

        self.random_masking_probability = random_masking_probability
        self.max_sequence_length = max_sequence_length
        self.code_shuffle = code_shuffle

        # check maximum sequence length
        max_length_seq_data = self.diagnosis_pd["ICD9_CODE"].map(lambda x: len(x)).max()
        ms_len = self.max_sequence_length
        logging.info(
            f"Max #codes per visit: {max_length_seq_data}, configured: {ms_len}, final: {ms_len + 1}"
        )

    def print_diagnosis_batch(self, batch_tokens: torch.Tensor, batch_y_visit: torch.Tensor):
        """
        Convert a batch of samples into output friendly format and print

        Parameters
        ----------
        batch_tokens: tensor holding rows of code ids
        batch_y_visit: tensor holding rows of target one-hot vectors
        """

        # for t, y1, y2 in zip(batch_tokens, batch_y_visit, batch_y_mask):
        for t, y1 in zip(batch_tokens, batch_y_visit):
            print("Input: ", " ".join(self.tokenizer.convert_ids_to_tokens(t.tolist())))

            y1_idx = np.where(y1 == 1)[0].tolist()
            print(
                "Target:",
                " ".join([self.tokenizer.disease_vocabulary.idx2word[c] for c in y1_idx]),
            )

            print()

    def compute_statistics(self):
        """Compute statistics over the `self.diagnosis_pd` backend data"""
        diagnosis_statistics(self.diagnosis_pd)

    def __len__(self):
        """Return dataset length"""
        return len(self.diagnosis_pd)

    def __getitem__(self, idx: int):
        """Get a single item from the dataset by `idx`"""

        # tokenize codes
        codes = self.diagnosis_pd["ICD9_CODE"].iloc[[idx]].values[0]
        code_ids = np.array(self.tokenizer.convert_tokens_to_ids(codes))

        # shuffle codes
        if self.code_shuffle:
            np.random.shuffle(code_ids)

        # create target vector multi-class visit target on CLS token
        target_visit = np.zeros(len(self.tokenizer.disease_vocabulary.word2idx))
        target_visit[[self.tokenizer.disease_vocabulary.word2idx[code] for code in codes]] = 1

        # random token masking
        if self.random_masking_probability > 0.0:
            code_ids = random_word_masking(
                code_ids,
                self.tokenizer.vocabulary,
                masking_probability=self.random_masking_probability,
            )

        # pad/cut sequence to max length
        code_ids = code_ids[: self.max_sequence_length]
        code_ids = np.pad(
            code_ids,
            (0, self.max_sequence_length - len(code_ids)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )

        # prepend CLS token
        input_ids = np.insert(code_ids, 0, self.cls_token_idx)

        item = (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(target_visit, dtype=torch.float32)
            # torch.tensor(target_masking, dtype=torch.float32)
        )

        return item


class CodeDataset(Dataset):
    """
    Dataset for graph pretraining using ICD and ATC codes

    Attributes
    ----------
    diagnosis_pd: DataFrame holding the ICD code source
    med_pd: DataFrame holding the ATC code source
    data_pd: DataFrame holding the combined codes
    tokenizer: the associated `CodeTokenizer`
    xxx_token_idx: ids for special tokens
    random_masking_probability: probability for random masking
        ignored if 0.0
    max_sequence_length: output gets cut/padded to this length
    code_shuffle: whether to shuffle codes upon loading
    """

    def __init__(
        self,
        diagnosis_file_path: str,
        prescription_file_path: str,
        prescription_mapping_directory: str,
        diagnosis_codes: Union[Set[str], Sequence[str]],
        prescription_codes: Union[Set[str], Sequence[str]],
        patient_ids: Union[Set[int], Sequence[int]] = None,
        code_count_range: Tuple[float, float] = (2, np.inf),
        visit_range: Tuple[float, float] = None,
        max_sequence_length: int = 47,
        random_masking_probability: float = 0.0,
        tokenizer: CodeTokenizer = None,
        code_shuffle: bool = True,
        notes_concepts_path: str = None,
        umls_graph_data: Dict = None,
        max_visit_text_length: int = 0,
        text_targets: bool = False,
        output_ids: bool = False,
        keep_no_med_patients: bool = False,
    ):
        """
        Constructor for `DiagnosisDataset`

        Parameters
        ----------
        diagnosis_file_path: path to source MIMIC-III `DIAGNOSES_ICD.csv`
        prescription_file_path: path to source MIMIC-III `PRESCRIPTIONS.csv`
        prescription_mapping_directory: directory holding `ndc2rxnorm_mapping.txt`
            and `ndc2atc_level4.csv` files for prescr. code mapping
        diagnosis_codes: set of disease codes used for dataset
        prescription_codes: set of prescriptions codes used for dataset
        patient_ids: list of patients to use in this dataset
        code_count_range: drop patients with a visit containing
            a code count outside of this range
        visit_range: only keep patients with number of visits within this range
        max_sequence_length: cut/pad output to this length
        random_masking_probability: probability for random masking,
        tokenizer: pass a `DiagnosisTokenizer`, will create one if None
        code_shuffle: shuffle visit codes for each load
        notes_concepts_path: str
            path to pre-extracted concepts for each
            note in MIMIC-III dataset
        umls_graph_data: Dict
            graph data and tokenizer over graph nodes
        max_visit_text_length: int
            maximum number of concept tokens per visit
            if 0, will take 95th percentile of data
        text_targets: bool
            output target vector for text concepts
        output_ids: bool
            whether to output the patient ids when loading samples
        keep_no_med_patients: bool
            keep patients without medication information in the dataset
        """
        super().__init__()

        self.keep_no_med_patients = keep_no_med_patients
        if keep_no_med_patients:
            logging.warning(f"[DATASET] keeping no med patients: {self.keep_no_med_patients}")

        # load disease codes source
        try:  # load cached
            self.diagnosis_pd = pd.read_pickle(
                path.join(
                    prescription_mapping_directory,
                    "preprocessed_mimic_tables/diagnoses_icd_preprocessed.pkl",
                )
            )
            logging.info(f"[ICD] loaded cached preprocessed shape: {self.diagnosis_pd.shape}")
        except:
            self.diagnosis_pd = process_diagnosis_table(diagnosis_file_path, uppercase_header=True)

        # only keep allowed patient records
        if patient_ids is not None:
            patient_ids = set(patient_ids)
            filter_lambda = lambda x: x["SUBJECT_ID"] in patient_ids
            self.diagnosis_pd = self.diagnosis_pd[self.diagnosis_pd.apply(filter_lambda, axis=1)]
            logging.info(f"[ICD] Filtered patient ids, filtered shape: {self.diagnosis_pd.shape}")

        logging.info(f"[CHECK 1] patient ids: {len(self.diagnosis_pd['SUBJECT_ID'].unique())}")

        # only keep code entries of provided code file
        self.diagnosis_pd = filter_by_visit_codes(self.diagnosis_pd, diagnosis_codes)

        logging.info(f"[CHECK 2] patient ids: {len(self.diagnosis_pd['SUBJECT_ID'].unique())}")

        try:  # load cached
            self.med_pd = pd.read_pickle(
                path.join(
                    prescription_mapping_directory,
                    "preprocessed_mimic_tables/prescriptions_atc_preprocessed.pkl",
                )
            )
            logging.info(f"[ATC] loaded cached preprocessed shape: {self.med_pd.shape}")

        except:
            # load prescription codes source
            self.med_pd = process_prescription_table(
                prescription_file_path, uppercase_header=True)

            # perform prescription code mapping
            ndc2rxnorm_mapping_file = path.join(
                prescription_mapping_directory, "ndc2rxnorm_mapping.txt"
            )
            rxnorm2atc_mapping_file = path.join(
                prescription_mapping_directory, "ndc2atc_level4.csv"
            )
            self.med_pd = map_ndc_atc4_codes(
                self.med_pd, ndc2rxnorm_mapping_file,
                rxnorm2atc_mapping_file, verbose=True
            )

        # filter by visit range
        if visit_range is not None:
            self.med_pd = filter_by_visit_range(self.med_pd, visit_range=visit_range)

        # match keys of diagnosis and prescription codes
        if self.keep_no_med_patients:
            if patient_ids is not None:
                patient_ids = set(patient_ids)
                filter_lambda = lambda x: x["SUBJECT_ID"] in patient_ids  # type: ignore
                self.med_pd = self.med_pd[self.med_pd.apply(filter_lambda, axis=1)]
                logging.info(f"[ATC] Filtered patient ids, filtered shape: {self.med_pd.shape}")
        else:
            self.diagnosis_pd, self.med_pd = match_diagnosis_prescriptions_ids(
                self.diagnosis_pd, self.med_pd
            )

        logging.info(f"[CHECK 3] patient ids: {len(self.diagnosis_pd['SUBJECT_ID'].unique())}")

        # group tables
        self.diagnosis_pd = group_table(self.diagnosis_pd, column="ICD9_CODE")
        self.med_pd = group_table(self.med_pd, column="ATC4")

        logging.info(f"[CHECK 4] patient ids: {len(self.diagnosis_pd['SUBJECT_ID'].unique())}")

        # filter patients number of remaining codes
        self.diagnosis_pd = filter_patients_number_codes(
            self.diagnosis_pd, code_range=code_count_range, code="ICD9_CODE"
        )
        self.med_pd = filter_patients_number_codes(
            self.med_pd, code_range=code_count_range, code="ATC4"
        )
        logging.info(f"[CHECK 5] patient ids: {len(self.diagnosis_pd['SUBJECT_ID'].unique())}")

        # merge tables
        join_type = "outer" if self.keep_no_med_patients else "inner"
        self.data_pd = merge_diagnosis_prescriptions(
            self.diagnosis_pd, self.med_pd, join_type=join_type
        )
        logging.info(f"[CHECK 6] patient ids: {len(self.data_pd['SUBJECT_ID'].unique())}")

        # impute nans caused by outer join if necessary
        if self.keep_no_med_patients:
            logging.info(f"[ATC] imputing patients without prescriptions")
            col_selector_med = self.data_pd["ATC4"].isna()
            self.data_pd.loc[col_selector_med, "ATC4"] = "[MASK]"
            self.data_pd.loc[col_selector_med, "ATC4"] = self.data_pd.loc[
                col_selector_med, "ATC4"
            ].map(lambda x: [x])

            col_selector_d = self.data_pd["ICD9_CODE"].isna()
            self.data_pd.loc[col_selector_d, "ICD9_CODE"] = "[MASK]"
            self.data_pd.loc[col_selector_d, "ICD9_CODE"] = self.data_pd.loc[
                col_selector_d, "ICD9_CODE"
            ].map(lambda x: [x])

        if tokenizer is None:
            self.tokenizer = CodeTokenizer(diagnosis_codes, prescription_codes)
        else:
            logging.info("Use existing tokenizer")
            self.tokenizer = tokenizer

        self.pad_token_idx = self.tokenizer.vocabulary.word2idx["[PAD]"]
        self.mask_token_idx = self.tokenizer.vocabulary.word2idx["[MASK]"]
        self.cls_token_idx = self.tokenizer.vocabulary.word2idx["[CLS]"]

        self.random_masking_probability = random_masking_probability
        self.max_sequence_length = max_sequence_length
        self.code_shuffle = code_shuffle
        self.output_ids = output_ids
        if self.output_ids:
            logging.info(f"[DATASET] outputs subject and visit ids")

        # check maximum sequence length
        max_length_seq_data_icd = self.diagnosis_pd["ICD9_CODE"].map(lambda x: len(x)).max()
        max_length_seq_data_atc = self.med_pd["ATC4"].map(lambda x: len(x)).max()
        ms_len = self.max_sequence_length
        logging.info(
            f"Max #codes per visit, ICD: {max_length_seq_data_icd}, ATC: {max_length_seq_data_atc}"
        )
        logging.info(f" > configured: {ms_len}, final: {ms_len + 1}")

        # Get UMLS graph and associated data
        self.umls_graph_data = umls_graph_data
        self.umls_tokenizer = None
        if self.umls_graph_data is not None:
            self.umls_tokenizer = self.umls_graph_data["tokenizer"]
            logging.info(f"[UMLS] UMLS graph data on input")

        # Load extracted note concepts
        self.notes_concepts = None
        if notes_concepts_path is not None:

            (tokenized_concepts, max_visit_text_length,) = load_and_preprocess_notes_concepts(
                notes_concepts_path=notes_concepts_path,
                umls_tokenizer=self.umls_tokenizer,  # type: ignore
                max_visit_text_length=max_visit_text_length,
                max_visit_text_length_perc=95,
                with_negation=True,
            )

            self.notes_concepts = tokenized_concepts
            self.max_visit_text_length = max_visit_text_length

        self.text_targets = text_targets
        if self.text_targets:
            assert self.notes_concepts is not None, "Need text input to use as target"
        logging.info(f"[DATASET] output text targets: {self.text_targets}")

    def get_max_concept_counts(self, percentile: int = 95):

        document_count_max = 0
        visit_count_max = 0
        visit_counts = []

        for patient in self.notes_concepts.keys():  # type: ignore
            for visit in self.notes_concepts[patient].keys():  # type: ignore
                visit_codes: Set[str] = set()
                visit_dict = self.notes_concepts[patient][visit]  # type: ignore

                for category in visit_dict:
                    for document in visit_dict[category]:
                        document_count_max = max(document_count_max, len(document))
                        visit_codes = visit_codes.union(document)

                visit_counts.append(len(visit_codes))
                visit_count_max = max(visit_count_max, len(visit_codes))

        visit_count_percentile = np.percentile(visit_counts, percentile)
        return document_count_max, visit_count_max, visit_count_percentile

    def print_code_batch(
        self,
        batch_t_d: torch.Tensor,
        batch_t_p: torch.Tensor,
        batch_y_d: torch.Tensor,
        batch_y_p: torch.Tensor,
        batch_text: torch.Tensor = None,
    ):
        """
        Convert a batch of samples into output friendly format and print

        Parameters
        ----------
        batch_t_d: tensor holding rows of ICD code ids
        batch_t_p: tensor holding rows of ATC code ids
        batch_y_X: tensors holding rows of target one-hot vectors
        """
        for i, (t_d, t_p, y_d, y_p) in enumerate(zip(batch_t_d, batch_t_p, batch_y_d, batch_y_p)):

            if self.umls_graph_data is None:
                print(
                    "Input ICD: ",
                    " ".join(self.tokenizer.convert_ids_to_tokens(t_d.tolist())),
                )
                print(
                    "Input ATC: ",
                    " ".join(self.tokenizer.convert_ids_to_tokens(t_p.tolist())),
                )
            else:
                print("Input ICD: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_d.tolist())))  # type: ignore
                print("Input ATC: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_p.tolist())))  # type: ignore

                if batch_text is not None:
                    print("Input Text: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(batch_text[i].tolist())))  # type: ignore

            yd_idx = np.where(y_d == 1)[0].tolist()
            yp_idx = np.where(y_p == 1)[0].tolist()
            print(
                "Target:",
                " ".join([self.tokenizer.disease_vocabulary.idx2word[c] for c in yd_idx]),
            )
            print(
                "Target:",
                " ".join([self.tokenizer.prescription_vocabulary.idx2word[c] for c in yp_idx]),
            )

            print()

    def compute_statistics(self, code="ICD9_CODE"):
        """
        Compute statistics over the `self.diagnosis_pd`
        and `self.med_pd` backend data
        """
        if code == "ICD9_CODE":
            diagnosis_statistics(self.diagnosis_pd, code="ICD9_CODE")
        else:
            diagnosis_statistics(self.med_pd, code="ATC4")

    def __len__(self):
        """Return dataset length"""
        return len(self.data_pd)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a single item from the dataset by `idx`"""

        # tokenize codes
        disease_codes = self.data_pd["ICD9_CODE"].iloc[[idx]].values[0]
        prescription_codes = self.data_pd["ATC4"].iloc[[idx]].values[0]

        if self.umls_tokenizer is not None:
            input_tokenizer = self.umls_tokenizer

            disease_codes_umls = [
                self.umls_graph_data["icd9_to_cui_map"][c]["cui"]  # type: ignore
                for c in disease_codes
            ]
            prescription_codes_umls = [
                self.umls_graph_data["atc_to_cui_map"][c]["cui"]  # type: ignore
                for c in prescription_codes
            ]

            disease_code_ids = np.array(
                self.umls_tokenizer.convert_tokens_to_ids(disease_codes_umls)
            )
            prescr_code_ids = np.array(
                self.umls_tokenizer.convert_tokens_to_ids(prescription_codes_umls)
            )

        else:
            input_tokenizer = self.tokenizer
            disease_code_ids = np.array(self.tokenizer.convert_tokens_to_ids(disease_codes))
            prescr_code_ids = np.array(self.tokenizer.convert_tokens_to_ids(prescription_codes))

        # shuffle codes
        if self.code_shuffle:
            np.random.shuffle(disease_code_ids)
            np.random.shuffle(prescr_code_ids)

        # create target vector multi-class visit target on CLS token
        target_visit_d = np.zeros(len(self.tokenizer.disease_vocabulary.word2idx))
        target_visit_d[
            [self.tokenizer.disease_vocabulary.word2idx[code] for code in disease_codes]
        ] = 1
        target_visit_p = np.zeros(len(self.tokenizer.prescription_vocabulary.word2idx))
        target_visit_p[
            [self.tokenizer.prescription_vocabulary.word2idx[code] for code in prescription_codes]
        ] = 1

        # random token masking
        if self.random_masking_probability > 0.0:
            disease_code_ids = random_word_masking(
                disease_code_ids,
                input_tokenizer.vocabulary,
                masking_probability=self.random_masking_probability,
            )
            prescr_code_ids = random_word_masking(
                prescr_code_ids,
                input_tokenizer.vocabulary,
                masking_probability=self.random_masking_probability,
            )

        # pad/cut sequence to max length
        disease_code_ids = disease_code_ids[: self.max_sequence_length]
        disease_code_ids = np.pad(
            disease_code_ids,
            (0, self.max_sequence_length - len(disease_code_ids)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )
        prescr_code_ids = prescr_code_ids[: self.max_sequence_length]
        prescr_code_ids = np.pad(
            prescr_code_ids,
            (0, self.max_sequence_length - len(prescr_code_ids)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )

        # prepend CLS token
        input_ids_d = np.insert(disease_code_ids, 0, self.cls_token_idx)
        input_ids_p = np.insert(prescr_code_ids, 0, self.cls_token_idx)

        # output if no text is added
        if self.notes_concepts is None:
            return (
                torch.tensor(input_ids_d, dtype=torch.int64),
                torch.tensor(input_ids_p, dtype=torch.int64),
                torch.tensor(target_visit_d, dtype=torch.float32),
                torch.tensor(target_visit_p, dtype=torch.float32),
            )

        # retrieve text concepts and
        # add to output

        # get patient/visit documents
        row = self.data_pd.iloc[idx]
        patient_id = int(row["SUBJECT_ID"])
        visit_id = int(row["HADM_ID"])

        # retrieve from pre-flattened and pre-tokenized
        # data dictionary
        try:
            text_codes_ids, negation_mask = self.notes_concepts[patient_id][visit_id]
        except KeyError:
            text_codes_ids = np.array([self.pad_token_idx])
            negation_mask = np.zeros(len(text_codes_ids), dtype=np.int32)


        # shuffle
        if self.code_shuffle:
            random_permuation = np.random.permutation(len(text_codes_ids))
            text_codes_ids = text_codes_ids[random_permuation]
            negation_mask = negation_mask[random_permuation]

        # cut
        text_codes_ids = text_codes_ids[: self.max_visit_text_length]
        negation_mask = negation_mask[: self.max_visit_text_length]

        # simpler efficient masking for text
        if self.random_masking_probability > 0.0:
            text_length = len(text_codes_ids)
            mask = np.random.binomial(n=1, p=self.random_masking_probability, size=text_length)
            text_codes_ids = np.where(mask == 1, self.mask_token_idx, text_codes_ids)

        # cut/pad
        text_codes_ids = np.pad(
            text_codes_ids,
            (0, self.max_visit_text_length - len(text_codes_ids)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )
        negation_mask = np.pad(
            negation_mask,
            (1, self.max_visit_text_length - len(negation_mask)),
            mode="constant",
            constant_values=0,
        )

        # prepend CLS token
        text_codes_ids = np.insert(text_codes_ids, 0, self.cls_token_idx)
        # `negation mask` has been accounted for in the previous padding step

        text_data = torch.stack(
            (
                torch.tensor(text_codes_ids, dtype=torch.int64),
                torch.tensor(negation_mask, dtype=torch.int64),
            )
        )

        # if we additionally compute text concept targets
        if self.text_targets:

            target_visit_text = np.zeros(len(self.umls_tokenizer.vocabulary.word2idx))
            try:
                text_codes_ids, negation_mask = self.notes_concepts[patient_id][visit_id]

                # create target vector multi-class visit target for text concept
                target_visit_text[text_codes_ids] = 1

            except KeyError:
                pass

            item = (
                torch.tensor(input_ids_d, dtype=torch.int64),
                torch.tensor(input_ids_p, dtype=torch.int64),
                text_data,
                torch.tensor(target_visit_d, dtype=torch.float32),
                torch.tensor(target_visit_p, dtype=torch.float32),
                torch.tensor(target_visit_text, dtype=torch.float32),
            )

        else:

            if self.output_ids:
                item = (
                    torch.tensor(input_ids_d, dtype=torch.int64),
                    torch.tensor(input_ids_p, dtype=torch.int64),
                    text_data,
                    torch.tensor(target_visit_d, dtype=torch.float32),
                    torch.tensor(target_visit_p, dtype=torch.float32),
                    (patient_id, visit_id),
                )  # type: ignore

            else:
                item = (
                    torch.tensor(input_ids_d, dtype=torch.int64),
                    torch.tensor(input_ids_p, dtype=torch.int64),
                    text_data,
                    torch.tensor(target_visit_d, dtype=torch.float32),
                    torch.tensor(target_visit_p, dtype=torch.float32),
                )  # type: ignore

        return item


# ===================================
#
# Datasets (CGL)
#
# ===================================
class CglDatasetDownstream(CodeDataset):
    """
    Dataset to reproduce and build upon
    results of the CGL: https://github.com/LuChang-CS/CGL
    repository and paper; used to benchmark disease tasks
    """

    def __init__(
        self,
        diagnosis_file_path: str,
        prescription_file_path: str,
        admissions_file_path: str,
        prescription_mapping_directory: str,
        diagnosis_codes: Union[Set[str], Sequence[str]],
        prescription_codes: Union[Set[str], Sequence[str]],
        patient_ids: Union[Set[int], Sequence[int]] = None,
        code_count_range: Tuple[float, float] = (2, np.inf),
        visit_range: Tuple[float, float] = None,
        max_sequence_length: int = 47,
        tokenizer: CodeTokenizer = None,
        code_shuffle: bool = True,
        target_task: str = "heart_failure",
        patient_sequence_length: int = 32,
        target_tokenizer: Optional[CodeTokenizer] = None,
        umls_graph_data: Dict = None,
        notes_concepts_path: str = None,
        max_visit_text_length: int = 0,
        eicu: bool = False,
        diagnosis_codes_target: Union[Set[str], Sequence[str]] = None,
        output_ids: bool = False,
    ):
        """
        Constructor for `CglDataset`

        Parameter
        ---------
        *: parameter description of `CodeDataset`
        admission_file_path: path to MIMIC-III ADMISSIONS.CSV
        target_task: {heart_failure, diagnosis}
        patient_sequence_length: maximal sequence length of admissions per patient
        target_tokenizer: target vocab tokenizer for the `diagnosis` task
        umls_graph_data: Dict
            dictioniary of data for a UMLS graph
            e.g. nx.Graph, tokenizer over nodes, ...
        notes_concepts_path: str
            path to pre-extracted concepts for each
            note in MIMIC-III dataset
        max_visit_text_length: int
            maximum number of concept tokens per visit
            if 0, will take 95th percentile of data
        eicu: bool
            whether we load eICU data
        diagnosis_codes_target: Union[Set[str], Sequence[str]]
            code set to consider for the targets
        output_ids: bool
            output patient ids
        """

        # initialize normal `CodeDataset`
        super().__init__(
            diagnosis_file_path=diagnosis_file_path,
            prescription_file_path=prescription_file_path,
            prescription_mapping_directory=prescription_mapping_directory,
            diagnosis_codes=diagnosis_codes,
            prescription_codes=prescription_codes,
            patient_ids=patient_ids,
            code_count_range=code_count_range,
            visit_range=visit_range,
            max_sequence_length=max_sequence_length,
            random_masking_probability=0.0,
            tokenizer=tokenizer,
            code_shuffle=code_shuffle,
            umls_graph_data=umls_graph_data,
            notes_concepts_path=notes_concepts_path,
            max_visit_text_length=max_visit_text_length,
            keep_no_med_patients=True,
            output_ids=output_ids,
        )

        self.eicu = eicu
        self.patient_sequence_length = patient_sequence_length
        self.max_sequence_length += 1
        if self.eicu:
            logging.info(f"[DATASET] load eICU data")

        # load target tokenizer for downstream task targets
        if target_tokenizer is None:
            if diagnosis_codes_target is None:
                target_codes = diagnosis_codes
            else:
                logging.info(f"[TARGET] Considering distinct target code set")
                target_codes = diagnosis_codes_target
            logging.info(f"[TARGET] Instatiating new target tokenizer, d: {len(target_codes)}")
            self.target_tokenizer = CodeTokenizer(target_codes, set(), special_tokens=[])
        else:
            logging.info("[TARGET] Use existing target tokenizer")
            self.target_tokenizer = target_tokenizer

        # set tokens
        self.target_pad_token_idx = self.tokenizer.vocabulary.word2idx["[PAD]"]
        self.target_mask_token_idx = self.tokenizer.vocabulary.word2idx["[MASK]"]
        self.target_cls_token_idx = self.tokenizer.vocabulary.word2idx["[CLS]"]

        self.target_task = target_task
        self.patient_data = self.transform_data_ordered(admissions_file_path)
        logging.info(
            f"Reformated data for visit sequence learning, num records: {len(self.patient_data)}"
        )
        logging.info(f"Maximum number of visits: {self.patient_sequence_length}")

    def transform_data_ordered(self, adm_file) -> List[Tuple]:
        """
        Transforms data from [admission, code] to
        [patient, admission, code] format order by admission time
        """
        logging.info(f"Reformating for visit sequence learning based on: {adm_file}")

        if self.eicu:
            admissions_pd = load_patient_table_eICU(adm_file)
            admissions_pd = admissions_pd.set_index(["SUBJECT_ID", "HADM_ID"])

            joined_pd = self.data_pd.copy().join(
                admissions_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner"
            )

            # if the offsets are positive we can sort them ascending
            # to achieve temporal ordering, default offsets are negative values where the
            # "largest" i.e. the smalles abs, is the first in the order
            # consider: https://eicu-crd.mit.edu/eicutables/patient/
            joined_pd["hospitaladmitoffset"] = joined_pd["hospitaladmitoffset"].abs()

        else:
            admissions_pd = pd.read_csv(
                adm_file,
                usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
                converters={"SUBJECT_ID": np.int64, "HADM_ID": np.int64, "ADMITTIME": str},
                index_col=["SUBJECT_ID", "HADM_ID"],
            )
            joined_pd = self.data_pd.copy().join(
                admissions_pd, on=["SUBJECT_ID", "HADM_ID"], how="inner"
            )

        records: List[Tuple] = []
        # iterate over all patients
        for subject_id in tqdm(joined_pd["SUBJECT_ID"].unique()):

            # extract all records for a patient
            item_df = joined_pd[joined_pd["SUBJECT_ID"] == subject_id]
            patient = []
            for _, row in item_df.iterrows():

                # get admission time
                if self.eicu:
                    admission_time = row["hospitaladmitoffset"]
                else:
                    admission_time = datetime.strptime(row["ADMITTIME"], "%Y-%m-%d %H:%M:%S")

                # without text note concepts
                if self.notes_concepts is None:
                    admission = (
                        list(row["ICD9_CODE"]),
                        list(row["ATC4"]),
                        admission_time,
                    )

                # with text
                else:

                    patient_id = int(row["SUBJECT_ID"])
                    visit_id = int(row["HADM_ID"])

                    # retrieve from preprocessed data
                    try:
                        text_codes = self.notes_concepts[patient_id][visit_id]

                    # this patient does not have any text notes
                    # in the provided data source
                    except KeyError:
                        text_codes_ids = np.array([self.pad_token_idx])
                        negation_mask = np.zeros(len(text_codes_ids), dtype=np.int32)
                        text_codes = (text_codes_ids, negation_mask)  # type: ignore

                    admission = (list(row["ICD9_CODE"]), list(row["ATC4"]), admission_time, text_codes)  # type: ignore

                # add to patient history
                patient.append(admission)

            # sort in-place by admission time
            patient.sort(key=lambda adm: adm[2])

            # skip patients without at least two admission
            if len(patient) < 2:
                continue

            records.append((subject_id, patient))

        return records

    def print_code_batch(
        self,
        batch_t_d: torch.Tensor,
        batch_t_p: torch.Tensor,
        batch_y,
        patients=1,
        batch_text: torch.Tensor = None,
    ):
        """
        Convert a batch of samples into output friendly format and print

        Parameters
        ----------
        batch_t_d: tensor holding rows of ICD code ids
        batch_t_p: tensor holding rows of ATC code ids
        batch_y_X: tensors holding rows of target one-hot vectors
        patients: subset of batch to print
        """
        batch_t_d = batch_t_d[:patients]
        batch_t_p = batch_t_p[:patients]
        batch_y = batch_y[:patients]

        for batch_i, (t_d, t_p, y) in enumerate(zip(batch_t_d, batch_t_p, batch_y)):
            print("===== Patient =====")
            for i in range(t_d.shape[0]):
                if self.umls_graph_data is None:
                    print(
                        "Input  ICD: ",
                        " ".join(self.tokenizer.convert_ids_to_tokens(t_d[i].tolist())),
                    )
                    print(
                        "Input  ATC: ",
                        " ".join(self.tokenizer.convert_ids_to_tokens(t_p[i].tolist())),
                    )
                else:
                    print("Input  ICD: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_d[i].tolist())))  # type: ignore
                    print("Input  ATC: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_p[i].tolist())))  # type: ignore

                    if batch_text is not None:
                        print("Input Text: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(batch_text[batch_i][i].tolist())))  # type: ignore

                if len(y[i].shape) == 0:  # 0-d tensor as target (scalar)
                    print(f"Target: {y[i]}")
                else:
                    y_idx = np.where(y[i] == 1)[0].tolist()
                    print(
                        "Target:",
                        " ".join(
                            sorted(
                                [
                                    self.target_tokenizer.disease_vocabulary.idx2word[c]
                                    for c in y_idx
                                ]
                            )
                        ),
                    )
            print("===== ======= =====\n")

    def __len__(self):
        """Return dataset length"""
        return len(self.patient_data)

    def get_target(self, disease_codes: Sequence[str]) -> Union[np.ndarray, int]:
        """
        Returns the target y according
        to the task `self.target_task` {heart_failure, diagnosis}

        Parameter
        ---------
        disease_codes: list of ICD9 codes
        """
        target_vocabulary = self.target_tokenizer.disease_vocabulary

        if self.target_task == "heart_failure":

            heart_failure = any(
                map(
                    lambda c: c.startswith(HEART_FAILURE_PREFIX)
                    and c in target_vocabulary.word2idx,
                    disease_codes,
                )
            )

            return 1 if heart_failure else 0

        if self.target_task == "diagnosis":

            target = np.zeros(len(target_vocabulary.word2idx))

            # pad at 0 indicates an invalid visit which will be masked
            # we return an empty target
            if disease_codes[0] == "[PAD]":
                return target

            target[
                [
                    target_vocabulary.word2idx[c]
                    for c in filter(lambda code: code in target_vocabulary.word2idx, disease_codes)
                ]
            ] = 1
            return target

        assert False, f"Unkown target task {self.target_task}"

    def __getitem__(self, idx: int):
        """Get a single item from the dataset by `idx`"""

        # get patient record
        patient_record = self.patient_data[idx][1]

        # tokenize codes using input tokenizer
        def default_convert_tokens(x, code=None):
            return self.tokenizer.convert_tokens_to_ids(x)

        def umls_convert_tokens(x, code="icd"):
            code_map = (
                self.umls_graph_data["icd9_to_cui_map"]
                if code == "icd"
                else self.umls_graph_data["atc_to_cui_map"]
            )
            mapper_func = lambda x: code_map[x]["cui"] if x in code_map else "[MASK]"
            codes = [mapper_func(c) for c in x]
            return self.umls_tokenizer.convert_tokens_to_ids(codes)

        convert_tokens = (
            default_convert_tokens if self.umls_graph_data is None else umls_convert_tokens
        )

        disease_codes = [admission[0] for admission in patient_record]
        prescription_codes = [admission[1] for admission in patient_record]
        disease_codes_ids = [
            np.array(convert_tokens(a, code="icd"), dtype=np.int64) for a in disease_codes
        ]
        prescr_codes_ids = [
            np.array(convert_tokens(a, code="atc"), dtype=np.int64) for a in prescription_codes
        ]

        # shuffle codes
        if self.code_shuffle:
            for i in range(len(disease_codes_ids)):
                np.random.shuffle(disease_codes_ids[i])
                np.random.shuffle(prescr_codes_ids[i])

        # prepend CLS
        disease_codes_ids = [np.insert(codes, 0, self.cls_token_idx) for codes in disease_codes_ids]
        prescr_codes_ids = [np.insert(codes, 0, self.cls_token_idx) for codes in prescr_codes_ids]

        # create target
        patient_pad_length = max(self.patient_sequence_length - len(disease_codes), 0)
        disease_codes_padded = disease_codes.copy()[: self.patient_sequence_length]
        disease_codes_padded.extend(
            [["[PAD]"] for _ in range(patient_pad_length)]
        )  # pad admissions
        target = list(map(lambda x: self.get_target(x), disease_codes_padded))  # map target

        # pad/cut sequence to max length
        def pad_code_length(
            codes,
            maxlen: int = self.max_sequence_length,
            constant_value: int = self.pad_token_idx,
        ):
            codes = codes[:maxlen]
            codes = np.pad(
                codes,
                (0, maxlen - len(codes)),
                mode="constant",
                constant_values=constant_value,
            )
            return codes

        disease_codes_ids = np.array(list(map(pad_code_length, disease_codes_ids)))  # type: ignore
        prescr_codes_ids = np.array(list(map(pad_code_length, prescr_codes_ids)))  # type: ignore

        disease_codes_ids = disease_codes_ids[: self.patient_sequence_length, :]  # type: ignore
        disease_codes_ids = np.pad(
            disease_codes_ids,
            ((0, self.patient_sequence_length - len(disease_codes_ids)), (0, 0)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )
        prescr_codes_ids = prescr_codes_ids[: self.patient_sequence_length, :]  # type: ignore
        prescr_codes_ids = np.pad(
            prescr_codes_ids,
            ((0, self.patient_sequence_length - len(prescr_codes_ids)), (0, 0)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )

        # assemble output tuple
        # output if no text is added
        if self.notes_concepts is None:
            item = (
                torch.tensor(disease_codes_ids, dtype=torch.int64),
                torch.tensor(prescr_codes_ids, dtype=torch.int64),
                torch.tensor(target, dtype=torch.float32),
            )

            return item

        # get text per visit
        text_codes_ids = list(map(lambda x: x[3][0], patient_record))
        negation_mask = list(map(lambda x: x[3][1], patient_record))

        # shuffle
        if self.code_shuffle:
            for i in range(len(text_codes_ids)):
                random_permuation = np.random.permutation(len(text_codes_ids[i]))
                text_codes_ids[i] = text_codes_ids[i][random_permuation]
                negation_mask[i] = negation_mask[i][random_permuation]

        # prepend CLS
        text_codes_ids = [np.insert(codes, 0, self.cls_token_idx) for codes in text_codes_ids]
        negation_mask = [np.insert(mask, 0, 0) for mask in negation_mask]

        # padding/cut on code level
        text_codes_ids = np.array(
            list(
                map(
                    lambda x: pad_code_length(x, maxlen=self.max_visit_text_length),
                    text_codes_ids,
                )
            )
        )  # type: ignore
        negation_mask = np.array(
            list(
                map(
                    lambda x: pad_code_length(
                        x, maxlen=self.max_visit_text_length, constant_value=0
                    ),
                    negation_mask,
                )
            )
        )  # type: ignore

        # padding on visit level
        text_codes_ids = text_codes_ids[: self.patient_sequence_length, :]  # type: ignore
        text_codes_ids = np.pad(
            text_codes_ids,
            ((0, self.patient_sequence_length - len(text_codes_ids)), (0, 0)),
            mode="constant",
            constant_values=self.pad_token_idx,
        )

        negation_mask = negation_mask[: self.patient_sequence_length, :]  # type: ignore
        negation_mask = np.pad(
            negation_mask,
            ((0, self.patient_sequence_length - len(negation_mask)), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        text_data = torch.stack(
            (
                torch.tensor(text_codes_ids, dtype=torch.int64),
                torch.tensor(negation_mask, dtype=torch.int64),
            )
        )

        if self.output_ids:
            patient_id = self.patient_data[idx][0]

            item = (  # type: ignore
                torch.tensor(disease_codes_ids, dtype=torch.int64),
                torch.tensor(prescr_codes_ids, dtype=torch.int64),
                text_data,
                torch.tensor(target, dtype=torch.float32),
                patient_id,
            )

        else:
            item = (  # type: ignore
                torch.tensor(disease_codes_ids, dtype=torch.int64),
                torch.tensor(prescr_codes_ids, dtype=torch.int64),
                text_data,
                torch.tensor(target, dtype=torch.float32),
            )

        return item
