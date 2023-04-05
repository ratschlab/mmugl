# ===============================================
#
# Data Processing Utilities
#
# Some methods have been adapted from: https://github.com/jshang123/G-Bert
# ===============================================
import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kg.data.vocabulary import SimpleTokenizer


# ========================================
#
# ICD codes
#
# ========================================
def process_diagnosis_table(file: str, uppercase_header: bool = False) -> pd.DataFrame:
    """
    Processes the MIMIC-III `DIAGNOSES_ICD.csv` table

    Parameters
    ----------
    file: path to input csv file
    uppercase_header: bool
        uppercase the column header

    Returns
    -------
    Pandas Dataframe holding the processed table data
    """
    logging.info("[ICD] Processing diagnosis table")

    diagnosis_pd = pd.read_csv(file)
    logging.info(f"[ICD] Table raw shape: {diagnosis_pd.shape}")

    if uppercase_header:
        diagnosis_pd.columns = diagnosis_pd.columns.str.upper()

    diagnosis_pd.dropna(inplace=True)  # drop rows with Nan values
    diagnosis_pd.drop(
        columns=["SEQ_NUM", "ROW_ID"], inplace=True
    )  # drop row_id and within admission seq. no
    diagnosis_pd.drop_duplicates(inplace=True)  # drop duplicates
    diagnosis_pd.sort_values(
        by=["SUBJECT_ID", "HADM_ID"], inplace=True
    )  # sort by patient and admission

    logging.info(f"[ICD] Table processed shape: {diagnosis_pd.shape}")
    return diagnosis_pd.reset_index(drop=True)


def filter_diagnosis_table(diagnosis_pd: pd.DataFrame, num: int = None) -> pd.DataFrame:
    """
    Filters out uncommon ICD9 codes, only keep top `num` codes

    Parameters
    ----------
    diagnosis_pd: Processed DataFrame of MIMIC-III `DIAGNOSES_ICD.csv`
    num: top-k codes to keep

    Returns
    -------
    DataFrame containing only top-k codes

    """
    if num is not None:
        logging.info(f"[ICD] Filter diagnosis to keep {num} most common codes")
        # compute count per code
        diagnosis_count = (
            diagnosis_pd.groupby(by=["ICD9_CODE"])
            .size()
            .reset_index()
            .rename(columns={0: "count"})
            .sort_values(by=["count"], ascending=False)
            .reset_index(drop=True)
        )

        # keep only `num` most common codes
        diagnosis_pd = diagnosis_pd[
            diagnosis_pd["ICD9_CODE"].isin(diagnosis_count.loc[:num, "ICD9_CODE"])
        ]

        logging.info(f"[ICD] Filtered table shape {diagnosis_pd.shape}")

        return diagnosis_pd.reset_index(drop=True)

    # else skip filtering
    else:
        return diagnosis_pd


def group_diagnosis_table(diagnosis_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Expects a flattened input table of patients and their codes and
    groups the codes for each patient and admission

    Parameters
    ----------
    diagnosis_pd: Processed DataFrame of MIMIC-III `DIAGNOSES_ICD.csv`

    Returns
    -------
    Table grouped by patient and admission with codes as lists
    """
    return group_table(diagnosis_pd, column="ICD9_CODE")


def filter_by_visit_codes(
    diagnosis_pd: pd.DataFrame, allowed_codes: Union[Set[str], Sequence[str]]
):
    """
    Filters out entries of MIMIC-III `DIAGNOSES_ICD.csv` to ensure only the
    codes in the vocabulary (given by `allowed_codes`) appear.

    Parameters
    ----------
    diagnosis_pd: processed DataFrame of MIMIC-III `DIAGNOSES_ICD.csv`
    allowed_codes: list of codes in vocabulary

    Returns
    -------
    Filtered DataFrame
    """

    allowed_codes = set(allowed_codes)
    temp = diagnosis_pd["ICD9_CODE"].map(lambda x: x in allowed_codes)

    diagnosis_pd = diagnosis_pd[temp]
    logging.info(
        f"[ICD] Remove non-vocabulary (size: {len(allowed_codes)}) codes, post: {diagnosis_pd.shape}"
    )

    return diagnosis_pd


def filter_patients_number_codes_icd(
    data_pd: pd.DataFrame, diagnosis_range: Tuple[float, float] = (2, np.inf)
) -> pd.DataFrame:
    """
    Filters out patients which have a visit with a code count outside of
    the given range by `diagnosis_range`. This operation is `inplace`.

    Parameters
    ----------
    data_pd: input DataFrame
    diagnosis_range: the allowed range of codes for a visit

    Returns
    -------
    The filtered DataFrame
    """
    return filter_patients_number_codes(data_pd, diagnosis_range, code="ICD9_CODE")


# ========================================
#
# NDC/ATC codes
#
# ========================================
def process_prescription_table(file: str, uppercase_header: bool = False) -> pd.DataFrame:
    """
    Loads and preprocesses the MIMIC-III `PRESCRIPTIONS.csv`

    Parameters
    ----------
    file: the path to the `PRESCRIPTIONS.csv` file
    uppercase_header: bool
        uppercase the column header
    """

    logging.info("[ATC] Processing prescriptions table")

    # assume that if we want to uppercase the stored
    # column names are lowercase, thus upon loading
    # we need to lowercase the column for which we fix the type
    ndc_column = "NDC"
    ndc_types = {ndc_column: "category", ndc_column.lower(): "category"}

    med_pd = pd.read_csv(file, dtype=ndc_types)
    logging.info(f"[ATC] Table raw shape: {med_pd.shape}")

    if uppercase_header:
        med_pd.columns = med_pd.columns.str.upper()

    # drop columns
    drop_columns = [
        "ROW_ID",
        "DRUG_TYPE",
        "DRUG_NAME_POE",
        "DRUG_NAME_GENERIC",
        "FORMULARY_DRUG_CD",
        "GSN",
        "PROD_STRENGTH",
        "DOSE_VAL_RX",
        "DOSE_UNIT_RX",
        "FORM_VAL_DISP",
        "FORM_UNIT_DISP",
        "FORM_UNIT_DISP",
        "ROUTE",
        "ENDDATE",
        "DRUG",
    ]
    med_pd.drop(columns=drop_columns, axis=1, inplace=True)

    # drop rows where NDC code is not available
    med_pd.drop(index=med_pd[med_pd["NDC"] == "0"].index, axis=0, inplace=True)

    # fill Nans take from last (table loads sorted by (SUBJECT, HADM))
    med_pd.fillna(method="pad", inplace=True)

    # drop remaining Nans (first few entries dropped) and duplicates
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)

    # reformat and type columns
    med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
    med_pd["STARTDATE"] = pd.to_datetime(med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S")
    med_pd = med_pd.reset_index(drop=True)

    logging.info(f"[ATC] Preprocessed shape: {med_pd.shape}")

    return med_pd


def map_ndc_atc4_codes(
    med_pd: pd.DataFrame, ndc2rxnorm_file: str, rxnorm2atc_file: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Maps the `NDC`-code column of a preprocessed MIMIC-III
    PRESCRIPTION.csv to `ATC4`-codes using provided mappings
    by the G-Bert repository:  https://github.com/jshang123/G-Bert

    Performs the mapping NDC -> RxNorm -> ATC4

    Parameters
    ----------
    med_pd: a preprocessed MIMIC-III PRESCRIPTIONS.csv
    ndc2rxnorm_file: path to a txt file containing a mapping dict
    rxnorm2atc_file: path to a csv file containing a mapping
    """
    rxnorm2atc = pd.read_csv(rxnorm2atc_file)
    with open(ndc2rxnorm_file, "r") as f:
        ndc2rxnorm = eval(f.read())

    # map ndc codes from MIMIC to RxNorm
    # RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/index.html
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    if verbose:
        logging.info(f"[ATC] Performed NDC -> RxNorm mapping, shape {med_pd.shape}")

    # map RxNorm codes to ATC4 codes
    rxnorm2atc = rxnorm2atc.drop(columns=["YEAR", "MONTH", "NDC"])  # drop unnecessary columns
    rxnorm2atc.drop_duplicates(
        subset=["RXCUI"], inplace=True
    )  # drop duplicate mapping codes in src
    med_pd.drop(
        index=med_pd[med_pd["RXCUI"].isin([""])].index, axis=0, inplace=True
    )  # drop empty rows

    # reset table
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.reset_index(drop=True)

    # map to ATC by merging on RxNorm codes
    med_pd = med_pd.merge(rxnorm2atc, on=["RXCUI"])
    med_pd.drop(columns=["NDC", "RXCUI"], inplace=True)  # drop unnecessary cols

    # reset table
    med_pd["ATC4"] = med_pd["ATC4"].map(lambda x: x[:5])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    logging.info(f"[ATC] Performed NDC -> ATC4 mapping, final shape {med_pd.shape}")

    return med_pd


# ========================================
#
# NOTES (NOTEEVENTS.csv)
#
# ========================================
def load_noteevents_table(file: str, drop_error: bool = True) -> pd.DataFrame:
    """
    Processes the MIMIC-III `NOTEEVENTS.csv` table

    Parameter
    ---------
    file: str
        path to input csv file
    drop_error: bool
        drop entries where `ISERROR` == 1
    """
    logging.info("[NOTE] processing note-events table")

    used_columns = [
        "SUBJECT_ID",
        "HADM_ID",
        "CATEGORY",
        "DESCRIPTION",
        "ISERROR",
        "TEXT",
    ]
    dtypes = {
        "SUBJECT_ID": pd.Int64Dtype(),
        "HADM_ID": pd.Int64Dtype(),
        "CATEGORY": str,
        "DESCRIPTION": str,
        "ISERROR": pd.Int16Dtype(),
        "TEXT": str,
    }
    noteevents_df = pd.read_csv(file, usecols=used_columns, dtype=dtypes)

    noteevents_df.drop_duplicates(inplace=True)  # drop duplicates
    noteevents_df.sort_values(["SUBJECT_ID", "HADM_ID"], inplace=True)

    # a 1 indicates an error, we thus keep the Nans
    if drop_error:
        noteevents_df = noteevents_df[noteevents_df["ISERROR"].isna()]
    noteevents_df.drop(columns=["ISERROR"], inplace=True)

    logging.info(f"[NOTE] processed table: {noteevents_df.shape}")
    return noteevents_df.reset_index(drop=True)


# ========================================
#
# GENERAL
#
# ========================================
def match_diagnosis_prescriptions_ids(
    diagnosis_pd: pd.DataFrame, med_pd: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keeps only (SUBJECT_ID, HADM_ID) entries appearing in both inputs

    Parameters
    ----------
    diagnosis_pd: preprocessed MIMIC-III DIAGNOSIS_ICD.csv
    med_pd: preprocessed MIMIC-III PRESCRIPTIONS.csv

    Returns
    -------
    A tuple of two DataFrames each only containing the matched entries
    """

    # extract key columns
    med_pd_key = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_pd_key = diagnosis_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()

    # merge to match keys (only keep common key-pairs)
    combined_key = med_pd_key.merge(diag_pd_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    # drop uncommon keys from the respective DataFrames
    diagnosis_pd = diagnosis_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    med_pd = med_pd.merge(combined_key, on=["SUBJECT_ID", "HADM_ID"], how="inner")

    logging.info(
        f"Match (SUBJECT_ID, HADM_ID), diagnosis: {diagnosis_pd.shape}, med: {med_pd.shape}"
    )

    return diagnosis_pd, med_pd


def merge_diagnosis_prescriptions(
    diagnosis_pd: pd.DataFrame, med_pd: pd.DataFrame, join_type: str = "inner"
) -> pd.DataFrame:
    """
    Merges a diagnosis code and prescription code DataFrame
    based on (SUBJECT_ID, HADM_ID)

    Parameters
    ----------
    diagnosis_pd: preprocessed MIMIC-III DIAGNOSIS_ICD.csv
    med_pd: preprocessed MIMIC-III PRESCRIPTIONS.csv
    join_type: str
        type of join to perform

    Returns
    -------
    A single merged dataframe
    """
    data_pd = diagnosis_pd.merge(med_pd, on=["SUBJECT_ID", "HADM_ID"], how=join_type)
    logging.info(f"Merged diagnosis and prescription codes, shape: {data_pd.shape}")

    return data_pd


def filter_patients_number_codes(
    data_pd: pd.DataFrame,
    code_range: Tuple[float, float] = (2, np.inf),
    code: str = "ICD9_CODE",
) -> pd.DataFrame:
    """
    Filters out patients which have a visit with a `code` count outside of
    the given range by `code_range`. This operation is `inplace`. Expects grouped inputs.

    Parameters
    ----------
    data_pd: input DataFrame
    code_range: the allowed range of codes for a visit
    code: the code to target

    Returns
    -------
    The filtered DataFrame
    """
    # compute number of codes per visit
    data_pd[f"{code}_Len"] = data_pd[f"{code}"].map(lambda x: len(x))

    # aggregate all visits and their code counts for each patient
    temp = (
        data_pd[["SUBJECT_ID", f"{code}_Len"]]
        .groupby(by="SUBJECT_ID")
        .agg({f"{code}_Len": lambda x: list(x)})
        .rename(columns={f"{code}_Len": f"{code}_Lens"})
    )

    # compute min/max number of codes in visits per patient
    temp[f"{code}_MaxLen"] = temp[f"{code}_Lens"].map(lambda x: max(x))
    temp[f"{code}_MinLen"] = temp[f"{code}_Lens"].map(lambda x: min(x))

    # filter based on `diagnosis_range`
    temp = temp[(temp[f"{code}_MaxLen"] > code_range[1]) | (temp[f"{code}_MinLen"] < code_range[0])]
    data_pd.drop(
        index=data_pd[data_pd["SUBJECT_ID"].isin(temp.index)].index,
        axis=0,
        inplace=True,
    )

    # drop helper column
    data_pd.drop(f"{code}_Len", inplace=True, axis=1)

    logging.info(
        f"Filtered patients to `{code}` count range {code_range}, filtered shape {data_pd.shape}"
    )

    return data_pd


def filter_by_visit_range(data_pd: pd.DataFrame, visit_range: Tuple[float, float] = (1, 2)):
    """
    Filters out patients with at least `visit_range[0]` visits and strictly less than
    `visit_range[1]` visists. `visit_range[0] <= num_visits < visit_range[1]`

    Parameters
    ----------
    data_pd: input data, expected to contain `SUBJECT_ID`, `HADM_ID` columns
    visit_range: allowed range of visits per patient

    Returns
    -------
    Filtered DataFrame according to `visit_range`
    """

    # group visits by subject
    temp = (
        data_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    # map to get unique visit count per patient
    temp["HADM_ID_Len"] = temp["HADM_ID"].map(lambda x: len(x))
    # filter out according to `visit_range`
    temp = temp[(temp["HADM_ID_Len"] >= visit_range[0]) & (temp["HADM_ID_Len"] < visit_range[1])]
    data_pd_filtered = temp.reset_index(drop=True)

    # filter by inner join on patients
    data_pd = data_pd.merge(data_pd_filtered[["SUBJECT_ID"]], on="SUBJECT_ID", how="inner")

    logging.info(f"Filtered visit range {visit_range}, final shape: {data_pd.shape}")

    return data_pd.reset_index(drop=True)


def split_dataset(
    data_pd: pd.DataFrame,
    allowed_patients: Sequence[int] = None,
    train: float = 0.7,
    val: float = 0.15,
    test: bool = False,
    random_state: int = 42,
    single_visit_train: bool = False,
) -> Tuple[Sequence[int], ...]:
    """
    Splits patient ids into train, validation (and test) sets

    Parameters
    ----------
    data_pd: input DataFrame
    allowed_patients: list of allowed patient ids to use for the splits
    train: fraction of data to use for training
    val: fraction of data to use for validation (only considered if `test`==True)
    test: boolean switch to create a test set
    random_state: set random state
    single_visit_train: use single visit patients only in training split

    Returns
    -------
    Returns a two/three tuple containing lists of patient ids
    for the respective splits
    """

    # retrieve all patient ids in given dataset
    all_patient_ids = set(data_pd["SUBJECT_ID"].unique())
    patient_ids = (
        all_patient_ids if allowed_patients is None else (all_patient_ids & set(allowed_patients))
    )

    logging.info(f"[SPLIT] considering {len(patient_ids)} patients")

    # keep single visit patients for training only
    if single_visit_train:
        single_visit_data_pd = filter_by_visit_range(data_pd.copy(), (1, 2))
        single_visit_patients = set(single_visit_data_pd["SUBJECT_ID"].unique())
        patient_ids = patient_ids - single_visit_patients
        logging.info(
            f"[SPLIT] keep {len(single_visit_patients)} single visit patients for training"
        )
        logging.info(f"[SPLIT] dividing remaining {len(patient_ids)} (multi-visit) into splits")
    else:
        single_visit_patients = set()

    train_ids, val_ids = train_test_split(
        list(patient_ids), random_state=random_state, test_size=1.0 - train
    )
    if test:
        not_train_size = 1.0 - train
        test_size = (not_train_size - val) * (1.0 / not_train_size)
        val_ids, test_ids = train_test_split(
            val_ids, random_state=random_state, test_size=test_size
        )

        train_ids = list(set.union(set(train_ids), single_visit_patients))

        logging.info(f"[SPLIT] train {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
        return train_ids, val_ids, test_ids

    else:
        train_ids = list(set.union(set(train_ids), single_visit_patients))
        logging.info(f"[SPLIT] train {len(train_ids)}, val: {len(val_ids)}")
        return train_ids, val_ids


def group_table(data_pd: pd.DataFrame, column: str = "ICD9_CODE") -> pd.DataFrame:
    """
    Expects a flattened input table of patients and their codes and
    groups the codes for each patient and admission

    Parameters
    ----------
    data_pd: Processed DataFrame of MIMIC-III  with (SUBJECT_ID, HADM_ID)
    column: column to group

    Returns
    -------
    Table grouped by patient and admission with codes as lists
    """

    # group codes per patient and admission
    data_pd = data_pd.groupby(by=["SUBJECT_ID", "HADM_ID"])[column].unique().reset_index()
    # map codes to be lists
    data_pd[column] = data_pd[column].map(lambda x: list(x))

    logging.info(f"Grouping codes {column} for (patients, admission), post shape {data_pd.shape}")

    return data_pd


def get_unique_codes(data_pd: pd.DataFrame, code="ICD9_CODE") -> Union[Set[Any], Sequence[Any]]:
    """
    Returns a set of all the unique codes in a
    grouped preprocessed MIMIC-III table

    Parameters
    ----------
    data_pd: grouped DataFrame of `DIAGNOSIS_ICD.csv` or `PRESCRIPTIONS.csv`
    code: column to extract unique members from

    Returns
    -------
    Set of unique members of column `code`
    """
    return set([j for i in data_pd[code].values for j in list(i)])


def write_set_codes(codes: Union[Sequence[str], Set[str]], path: str):
    """
    Write a set of `codes` i.e. str to a file at `path`

    Parameters
    ----------
    codes: the set of codes to write to file
    path: file path
    """
    logging.info(f"Write {len(codes)} codes to file {path}")
    output_file = open(path, "w")
    list(map(lambda x: output_file.write(x + "\n"), codes))
    output_file.close()


def read_set_codes(path: str) -> List[str]:
    """
    Reads a set of `codes` from file and returns a list (keep order)

    Parameters
    ----------
    path: path to input file

    Returns
    -------
    Set of code strings
    """
    # open file
    code_file = open(path, "r")
    code_file_content = code_file.read()

    # parse content
    code_list = code_file_content.split("\n")
    code_file.close()

    # remove empty codes
    code_set = list(code_list)
    code_set.remove("")

    return code_set


# ===================================
#
# Text tokens
#
# ===================================
def get_max_concept_counts(notes_concepts, percentile: int = 95):
    """
    Compute percentile of token counts
    per visit in a loaded notes_concepts dictionary
    """

    document_count_max = 0
    visit_count_max = 0
    visit_counts = []

    for patient in notes_concepts.keys():  # type: ignore
        for visit in notes_concepts[patient].keys():  # type: ignore
            visit_codes: Set[str] = set()
            visit_dict = notes_concepts[patient][visit]  # type: ignore

            for category in visit_dict:
                for document in visit_dict[category]:
                    document_count_max = max(document_count_max, len(document))
                    visit_codes = visit_codes.union(document)

            visit_counts.append(len(visit_codes))
            visit_count_max = max(visit_count_max, len(visit_codes))

    visit_count_percentile = np.percentile(visit_counts, percentile)
    return document_count_max, visit_count_max, visit_count_percentile


def load_and_preprocess_notes_concepts(
    notes_concepts_path: str,
    umls_tokenizer: SimpleTokenizer,
    max_visit_text_length: int = 0,
    max_visit_text_length_perc: int = 95,
    with_negation: bool = False,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], int]:
    """
    Preprocess a notes concepts dictionary
    - load it
    - compute desired max text token length
            by percentile
    - flatten the tokens
    - tokenize them

    Parameter
    ---------
    notes_concepts_path: str
            path to load the notes concepts from
    max_visit_text_length: int
            maximum text length, if 0, will compute
            `max_visit_text_length_perc`th percentile
    max_visit_text_length_perc: int
            percentile to compute of max_visit_text_length
    with_negation: bool
            if the source file contains negation information
            which should be extracted
    """

    # Load data
    logging.info(f"[CONCEPTS] Loading note concepts from: {notes_concepts_path}")
    with open(notes_concepts_path, "rb") as file:
        notes_concepts = pickle.load(file)
    logging.info(f"[CONCEPTS] {len(notes_concepts)} patients")

    # Compute max visit text length percentile
    if max_visit_text_length == 0:
        percentile = max_visit_text_length_perc
        doc_max, visit_max, visit_perc = get_max_concept_counts(
            notes_concepts, percentile=percentile
        )

        logging.info(f"[CONCEPTS] document max length: {doc_max}")
        logging.info(f"[CONCEPTS] visit max length: {visit_max}")
        logging.info(f"[CONCEPTS] visit {percentile}th percentile length: {visit_perc}")
        max_visit_text_length = int(visit_perc)

    logging.info(f"[CONCEPTS] maximum visit text length: {max_visit_text_length}")

    # Flatten and tokenize text concepts
    logging.info(f"[CONCEPTS] Flattening and tokenizing notes text concepts")
    tokenized_concepts: Dict = {}
    for patient_id, source_patient_dict in tqdm(notes_concepts.items()):

        # stores all visits of a patient by visit_id (HADM_ID)
        tokenized_concepts[patient_id] = {}

        for visit_id, visit_categories in source_patient_dict.items():

            # flatten codes in all categories and documents
            # remove duplicates by set creation
            negation_mask = None
            if with_negation:
                flat_text_codes_ext = list(
                    set(
                        [
                            (code[0], code[2])
                            for documents in visit_categories.values()
                            for document in documents
                            for code in document
                        ]
                    )
                )
                flat_text_codes = list(map(lambda x: x[0], flat_text_codes_ext))
                negation_mask = np.array(
                    list(map(lambda x: x[1], flat_text_codes_ext)), dtype=np.int32
                )
                # negation_mask = np.where(negation_mask == -1, 0, 1) # 0: negated, 1: positive
            else:
                flat_text_codes = list(
                    set(
                        [
                            code[0]
                            for documents in visit_categories.values()
                            for document in documents
                            for code in document
                        ]
                    )
                )

            # tokenize
            text_codes_ids = np.array(
                umls_tokenizer.convert_tokens_to_ids(flat_text_codes), dtype=np.int64
            )

            # store tokenized ids
            tokenized_concepts[patient_id][visit_id] = (
                (text_codes_ids, negation_mask) if with_negation else text_codes_ids
            )
    logging.info(f"[CONCEPTS] Pretokenized notes concepts for {len(tokenized_concepts)} patients")

    return tokenized_concepts, max_visit_text_length
