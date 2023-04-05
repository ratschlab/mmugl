# ===============================================
#
#   Torch Cached Datasets for Training
#   Loading prepared datasets from disk
#
# ===============================================
import logging
import pickle
import random
from datetime import datetime
from os import path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from kg.data.datasets import CodeDataset
from kg.data.processing import load_and_preprocess_notes_concepts, read_set_codes
from kg.data.vocabulary import CodeTokenizer


class GbertDataset(CodeDataset):
    """
    Dataset to load and work with the provided data
    from: https://github.com/jshang123/G-Bert
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        pretraining: bool = True,
        tokenizer: Optional[CodeTokenizer] = None,
        max_sequence_length: int = 47,
        random_masking_probability: float = 0.0,
        code_shuffle: bool = True,
        notes_concepts_path: str = None,
        umls_graph_data: Dict = None,
        max_visit_text_length: int = 0,
    ):
        """
        Constructor for `GbertDataset`

        Parameters
        ----------
        data_dir: dir holding the provided files from G-Bert repository
        split: {train, val, test}
        pretraining: whether to prepare the data for pretraining or downstream
            training according to G-Bert data preparation
        tokenizer: input tokenizer to use
        max_sequence_length: -
        random_masking_probability: -
        code_shuffle: -
        notes_concepts_path: str
            path to pre-extracted concepts for each
            note in MIMIC-III dataset
        umls_graph_data: Dict
            dictioniary of data for a UMLS graph
            e.g. nx.Graph, tokenizer over nodes, ...
        max_visit_text_length: int
            maximum number of concept tokens per visit
            if 0, will take 95th percentile of data
        """
        logging.info("[G-Bert] loading provided G-Bert dataset")
        self.data_dir = data_dir

        # load multi visit data and filter by patient id
        data_multi = pd.read_pickle(path.join(data_dir, "data-multi-visit.pkl"))
        logging.info(f"Loaded multi visit data, shape: {data_multi.shape}")

        # load patient ids for split
        split_id_file_map = {
            "train": "train-id.txt",
            "val": "eval-id.txt",
            "test": "test-id.txt",
        }
        patient_ids = self.load_ids(path.join(data_dir, split_id_file_map[split]))
        logging.info(f"Loaded {len(patient_ids)} patient ids for split {split}")

        # filter out patients according to split
        patient_ids = set(patient_ids)  # type: ignore
        filter_lambda = lambda x: x["SUBJECT_ID"] in patient_ids
        data_multi = data_multi[data_multi.apply(filter_lambda, axis=1)]
        logging.info(f"Filtered patient ids, filtered shape: {data_multi.shape}")

        # load single visit data (not filter by provided ids as they are
        #   separate patients filtered upon dataset creation)
        if pretraining and split == "train":
            logging.info(f"Joining in single visit data for pretraining")
            data_single = pd.read_pickle(path.join(data_dir, "data-single-visit.pkl"))
            self.data_pd = pd.concat([data_single, data_multi]).reset_index(drop=True)
        else:
            self.data_pd = data_multi
        logging.info(f"Final data, shape: {self.data_pd.shape}")

        # sanity check ids
        self.check_ids(split, split_id_file_map)

        # load tokenizer
        if tokenizer is None:
            d_codes = read_set_codes(path.join(data_dir, "dx-vocab.txt"))
            p_codes = read_set_codes(path.join(data_dir, "rx-vocab.txt"))
            logging.info(f"Instatiating new tokenizer, d: {len(d_codes)}, p: {len(p_codes)}")
            self.tokenizer = CodeTokenizer(
                diagnosis_codes=d_codes,
                prescription_codes=p_codes,
            )
        else:
            logging.info(f"Use existing tokenizer, size: {len(tokenizer.vocabulary.idx2word)}")
            self.tokenizer = tokenizer

        # set tokens
        self.pad_token_idx = self.tokenizer.vocabulary.word2idx["[PAD]"]
        self.mask_token_idx = self.tokenizer.vocabulary.word2idx["[MASK]"]
        self.cls_token_idx = self.tokenizer.vocabulary.word2idx["[CLS]"]

        # set variables
        self.diagnosis_pd = pd.DataFrame(self.data_pd[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]])
        self.med_pd = pd.DataFrame(self.data_pd[["SUBJECT_ID", "HADM_ID", "ATC4"]])

        self.max_sequence_length = max_sequence_length
        self.code_shuffle = code_shuffle
        self.random_masking_probability = random_masking_probability

        # check maximum sequence length
        max_length_seq_data_icd = self.diagnosis_pd["ICD9_CODE"].map(lambda x: len(x)).max()
        max_length_seq_data_atc = self.med_pd["ATC4"].map(lambda x: len(x)).max()
        ms_len = self.max_sequence_length
        logging.info(
            f"Max #codes per visit, ICD: {max_length_seq_data_icd}, ATC: {max_length_seq_data_atc}"
        )
        logging.info(f" > configured: {ms_len}, final: {ms_len + 1}")

        # get UMLS graph data
        self.umls_graph_data = umls_graph_data
        self.umls_tokenizer = None
        if self.umls_graph_data is not None:
            self.umls_tokenizer = self.umls_graph_data["tokenizer"]
            logging.info(f"[UMLS] Using UMLS graph data on input")

        # Load extracted note concepts
        self.notes_concepts = None
        if notes_concepts_path is not None:

            (tokenized_concepts, max_visit_text_length,) = load_and_preprocess_notes_concepts(
                notes_concepts_path=notes_concepts_path,
                umls_tokenizer=self.umls_tokenizer,
                max_visit_text_length=max_visit_text_length,
                max_visit_text_length_perc=95,
                with_negation=True,
            )

            self.notes_concepts = tokenized_concepts
            self.max_visit_text_length = max_visit_text_length

        self.text_targets = False
        self.output_ids = False

    @staticmethod
    def load_ids(id_file: str) -> List[int]:
        """Loads a list of ids from a file"""
        ids = []
        with open(id_file, "r") as f:
            for line in f:
                ids.append(int(line.rstrip("\n")))

        return ids

    def check_ids(self, split: str, file_map: Dict[str, str]):
        """
        If split on `train` checks none of the validation and test
        ids are in the training set.
        If overlap is detected the violating ids are removed from train

        Parameter
        ---------
        split: the training split of this dataset
        file_map: from split to file name for loading
        """

        if split != "train":
            return

        logging.info("[G-Bert] checking id overlap for train/test")

        val_ids = self.load_ids(path.join(self.data_dir, file_map["val"]))
        test_ids = self.load_ids(path.join(self.data_dir, file_map["test"]))
        forbidden_ids = set(val_ids + test_ids)

        train_ids = set(set(self.data_pd["SUBJECT_ID"].unique()))

        intersection = train_ids.intersection(forbidden_ids)
        overlap = len(intersection)

        if overlap > 0:
            logging.warning(f"Detected patient overlap between train/(val, test) of {overlap}")

            filter_lambda = lambda x: x["SUBJECT_ID"] not in forbidden_ids
            self.data_pd = self.data_pd[self.data_pd.apply(filter_lambda, axis=1)]
            logging.info(f"Dropped overlap, final dataset shape: {self.data_pd.shape}")


class GbertDatasetDownstream(GbertDataset):
    """
    Dataset to load and work with the provided data
    from: https://github.com/jshang123/G-Bert; adjusted
    __getitem__ dataloading for downstream tasks learning
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tokenizer: Optional[CodeTokenizer] = None,
        target_tokenizer: Optional[CodeTokenizer] = None,
        max_sequence_length: int = 47,
        patient_sequence_length: int = 32,
        code_shuffle: bool = True,
        admissions_file_path: Optional[str] = None,
        notes_concepts_path: str = None,
        umls_graph_data: Dict = None,
        max_visit_text_length: int = 0,
    ):
        """
        Constructor for `GbertDatasetDownstream`

        Parameters
        ----------
        data_dir: dir holding the provided files from G-Bert repository
        split: {train, val, test}
        tokenizer: input tokenizer to use
        target_tokenizer: tokenizer for the targets
        max_sequence_length: sequence length for the codes
        patient_sequence_length: max sequence length for admissions per patient
        code_shuffle: -
        notes_concepts_path: str
            Path to load the extracted concepts for
            each patient
        umls_graph_data: Dict
            dictioniary of data for a UMLS graph
            e.g. nx.Graph, tokenizer over nodes, ...
        """

        # load parent class
        super().__init__(
            data_dir=data_dir,
            split=split,
            pretraining=False,
            tokenizer=tokenizer,
            max_sequence_length=max_sequence_length,
            random_masking_probability=0.0,
            code_shuffle=code_shuffle,
            umls_graph_data=umls_graph_data,
            notes_concepts_path=notes_concepts_path,
            max_visit_text_length=max_visit_text_length,
        )

        # set vars
        self.patient_sequence_length = patient_sequence_length
        logging.info(f"Patient sequence length (num visits): {patient_sequence_length}")
        self.max_sequence_length += 1

        # load target tokenizer for downstream task targets
        if target_tokenizer is None:
            d_codes = read_set_codes(path.join(data_dir, "dx-vocab-multi.txt"))
            p_codes = read_set_codes(path.join(data_dir, "rx-vocab-multi.txt"))
            logging.info(f"Instatiating new target tokenizer, d: {len(d_codes)}, p: {len(p_codes)}")
            self.target_tokenizer = CodeTokenizer(d_codes, p_codes)
        else:
            logging.info("Use existing target tokenizer")
            self.target_tokenizer = target_tokenizer

        # set tokens
        self.target_pad_token_idx = self.tokenizer.vocabulary.word2idx["[PAD]"]
        self.target_mask_token_idx = self.tokenizer.vocabulary.word2idx["[MASK]"]
        self.target_cls_token_idx = self.tokenizer.vocabulary.word2idx["[CLS]"]

        # rearange data for [patient, admissions, code] format
        if admissions_file_path is not None:
            logging.info("Reformating data ordered by admission time")
            self.patient_data = self.transform_data_ordered(admissions_file_path)
        else:
            logging.error("Loading patient data not strictly ordered")
            self.patient_data = self.transform_data()

        logging.info(
            f"Reformated data for visit sequence learning, num records: {len(self.patient_data)}"
        )

    def transform_data(self):
        """Transforms data from [admission, code] to [patient, admission, code] format"""
        records = []
        # iterate over all patients
        for subject_id in self.data_pd["SUBJECT_ID"].unique():

            # extract all records for a patient
            item_df = self.data_pd[self.data_pd["SUBJECT_ID"] == subject_id]
            patient = []
            for _, row in item_df.iterrows():
                admission = [list(row["ICD9_CODE"]), list(row["ATC4"])]
                patient.append(admission)

            # skip patients without at least two admission
            if len(patient) < 2:
                continue
            records.append((subject_id, patient))

        return records

    def transform_data_ordered(self, adm_file):
        """
        Transforms data from [admission, code] to
        [patient, admission, code] format order by time
        """

        admissions_pd = pd.read_csv(
            adm_file,
            usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
            converters={"SUBJECT_ID": np.int, "HADM_ID": np.int, "ADMITTIME": np.str},
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
        batch_y_d: torch.Tensor,
        batch_y_p: torch.Tensor,
        patients=1,
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
        batch_y_d = batch_y_d[:patients]
        batch_y_p = batch_y_p[:patients]

        for t_d, t_p, y_d, y_p in zip(batch_t_d, batch_t_p, batch_y_d, batch_y_p):
            print("===== Patient =====")
            for i in range(t_d.shape[0]):
                if self.umls_graph_data is None:
                    print(
                        "Input ICD: ",
                        " ".join(self.tokenizer.convert_ids_to_tokens(t_d[i].tolist())),
                    )
                    print(
                        "Input ATC: ",
                        " ".join(self.tokenizer.convert_ids_to_tokens(t_p[i].tolist())),
                    )
                else:
                    print("Input ICD: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_d[i].tolist())))  # type: ignore
                    print("Input ATC: ", " ".join(self.umls_tokenizer.convert_ids_to_tokens(t_p[i].tolist())))  # type: ignore

                yd_idx = np.where(y_d[i] == 1)[0].tolist()
                yp_idx = np.where(y_p[i] == 1)[0].tolist()
                print(
                    "Target:",
                    " ".join(
                        sorted(
                            [self.target_tokenizer.disease_vocabulary.idx2word[c] for c in yd_idx]
                        )
                    ),
                )
                print(
                    "Target:",
                    " ".join(
                        sorted(
                            [
                                self.target_tokenizer.prescription_vocabulary.idx2word[c]
                                for c in yp_idx
                            ]
                        )
                    ),
                )
                print()
            print("===== ======= =====\n")

    def __len__(self):
        """Return dataset length"""
        return len(self.patient_data)

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
            codes = [code_map[c]["cui"] for c in x]
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

        # create target vector multi-class visit target on CLS token
        # use target tokenizer
        target_visit_d = np.zeros(
            (
                self.patient_sequence_length,
                len(self.target_tokenizer.disease_vocabulary.word2idx),
            )
        )
        for i, codes in enumerate(disease_codes):
            target_visit_d[
                i, [self.target_tokenizer.disease_vocabulary.word2idx[c] for c in codes]
            ] = 1

        target_visit_p = np.zeros(
            (
                self.patient_sequence_length,
                len(self.target_tokenizer.prescription_vocabulary.word2idx),
            )
        )
        for i, codes in enumerate(prescription_codes):
            target_visit_p[
                i,
                [self.target_tokenizer.prescription_vocabulary.word2idx[c] for c in codes],
            ] = 1

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

        if self.notes_concepts is None:
            item = (
                torch.tensor(disease_codes_ids, dtype=torch.int64),
                torch.tensor(prescr_codes_ids, dtype=torch.int64),
                # torch.tensor(target_visit_d, dtype=torch.float32),
                torch.tensor(target_visit_p, dtype=torch.float32),
            )

            return item

        # --------------------------
        # with text
        # --------------------------

        # get pre-tokenized text per visit
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

        item = (  # type: ignore
            torch.tensor(disease_codes_ids, dtype=torch.int64),
            torch.tensor(prescr_codes_ids, dtype=torch.int64),
            text_data,
            torch.tensor(target_visit_p, dtype=torch.float32),
        )

        return item
