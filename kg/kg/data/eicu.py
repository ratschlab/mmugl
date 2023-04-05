# ===============================================
#
# Data Processing Utilities
# for the eICU dataset
#
# ===============================================
import logging
import multiprocessing as mp
from functools import partial
from os import path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests  # type: ignore
from quickumls import QuickUMLS
from tqdm import tqdm

from kg.data.quickumls import QUSettings
from kg.data.umls import load_mrconso
from kg.utils.constants import QUICKUMLS_ACCEPTED_SEMTYPES


def load_patient_table_eICU(patients_filepath: str) -> pd.DataFrame:
    """
    Load the eICU patient table

    Parameter
    ---------
    patients_filepath: str
        path to the eICU patient table HDF5 file

    Return
    ------
    patients_pd:
    """
    patient_pd = pd.read_hdf(
        patients_filepath,
        columns=["patientunitstayid", "patienthealthsystemstayid", "hospitaladmitoffset"],
    )

    # rename to match to MIMIC-III naming
    # we treat each hospital stay as an individual "patient"
    # and each ICU stay as a "visit" to perform time-series
    # modeling
    rename_dict = {
        "patientunitstayid": "HADM_ID",  # unit stay -> 'visit'
        "patienthealthsystemstayid": "SUBJECT_ID",  # visit -> 'patient'
    }
    patient_pd = patient_pd.rename(columns=rename_dict)

    return patient_pd


def process_diagnosis_table_eICU(diagnosis_filepath: str, patient_filepath: str) -> pd.DataFrame:
    """
    Process diagnosis table from eICU

    Parameter
    ---------
    diagnosis_filepath: str
        path to diagnosis.h5 from eICU
    patient_filepath: str
        path to patient.h5 from eICU

    Returns
    -------
    diagnosis_pd: pd.DataFrame
        DF with ICD9 codes and visit, subject identifiers
    """
    logging.info("[ICD] Processing diagnosis table")

    keep_columns = ["diagnosisid", "patientunitstayid", "icd9code"]
    diagnosis_pd = pd.read_hdf(diagnosis_filepath, columns=keep_columns)
    diagnosis_pd = diagnosis_pd.set_index(keys="diagnosisid")

    logging.info(f"[ICD] Table raw shape: {diagnosis_pd.shape}")

    diagnosis_pd.dropna(inplace=True)  # drop rows with Nan values
    diagnosis_pd.drop_duplicates(inplace=True)  # drop duplicates

    allowed_initial_letters: Set[str] = set()
    allowed_initial_letters.update(map(str, range(0, 10)))
    allowed_initial_letters.update(["E", "V"])

    # extract/transform ICD codes
    def extract_icd(row: str) -> Optional[str]:
        # consider only columns with both ICD9 and ICD10
        if "," not in row:
            return None

        # get ICD-9 code
        row = row.split(",")[0]
        row = row.replace(".", "")  # remove dots

        if row[0:1] not in allowed_initial_letters:
            return None

        return row

    diagnosis_pd["icd9code"] = diagnosis_pd["icd9code"].map(extract_icd)

    diagnosis_pd.dropna(inplace=True)  # drop rows with Nan values
    logging.info(f"[ICD] mapped and dropped Nans: {diagnosis_pd.shape}")

    logging.info(f"[ICD] Load and merge patient table")
    patient_table = pd.read_hdf(
        patient_filepath, columns=["patientunitstayid", "patienthealthsystemstayid"]
    )
    diagnosis_pd = diagnosis_pd.merge(patient_table, on="patientunitstayid")

    # rename to match to MIMIC-III naming
    # we treat each hospital stay as an individual "patient"
    # and each ICU stay as a "visit" to perform time-series
    # modeling
    rename_dict = {
        "icd9code": "ICD9_CODE",
        "patientunitstayid": "HADM_ID",  # unit stay -> 'visit'
        "patienthealthsystemstayid": "SUBJECT_ID",  # visit -> 'patient'
    }
    diagnosis_pd = diagnosis_pd.rename(columns=rename_dict)

    logging.info(f"[ICD] Table processed shape: {diagnosis_pd.shape}")
    return diagnosis_pd


def process_note_table_eICU(notes_filepath: str, patient_filepath: str) -> pd.DataFrame:
    """
    Process note table from eICU, concat all reasonable fields into a single
    string for further processing and treatment as text note

    Parameter
    ---------
    notes_filepath: str
        path to note.h5 from eICU
    patient_filepath: str
        path to patient.h5 from eICU

    Returns
    -------
    note_pd: pd.DataFrame
        DF with concatenated text information, visit, subject identifiers
    """
    logging.info(f"[NOTE] Processing notes table")
    note_pd = pd.read_hdf(
        notes_filepath, columns=["patientunitstayid", "notetype", "notevalue", "notetext"]
    )
    logging.info(f"[NOTE] Table raw shape: {note_pd.shape}")

    note_pd["TEXT"] = str(
        note_pd["notetype"] + " " + note_pd["notevalue"] + " " + note_pd["notetext"]
    )
    note_pd["CATEGORY"] = "eICU"

    note_pd.drop(columns=["notetype", "notevalue", "notetext"], inplace=True)
    note_pd.drop_duplicates(inplace=True)  # drop duplicates

    logging.info(f"[NOTE] Load and merge patient table")
    patient_table = pd.read_hdf(
        patient_filepath, columns=["patientunitstayid", "patienthealthsystemstayid"]
    )
    note_pd = note_pd.merge(patient_table, on="patientunitstayid")

    rename_dict = {
        "patientunitstayid": "HADM_ID",  # unit stay -> 'visit'
        "patienthealthsystemstayid": "SUBJECT_ID",  # visit -> 'patient'
    }
    note_pd = note_pd.rename(columns=rename_dict)

    logging.info(f"[NOTE] Table processed shape: {note_pd.shape}")

    return note_pd


def rxnorm_to_atc_api(
    rxnorm: str,
    verbose: bool = False,
) -> str:
    """
    Perform an API call to convert
    an RxNorm id to an ATC code

    Parameter
    ---------
    rxnorm: str
        rxnorm id as a string

    Return
    ------
    atc_code: str
        atc code corresponding to input rxnorm
    """

    # Perform conversion to ATC using API
    # base_url = 'https://rxnav.nlm.nih.gov/REST/'
    atc_url = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui="

    proxies = {
        "http": "http://localhost:29488",
        "https": "http://localhost:29488",
    }

    try:
        url = atc_url + rxnorm
        response = requests.get(url, proxies=proxies).json()

        matches = list(
            filter(
                lambda x: x["relaSource"] == "ATC",
                response["rxclassDrugInfoList"]["rxclassDrugInfo"],
            )
        )
        atc4_code = matches[0]["rxclassMinConceptItem"]["classId"]  # [0:4]

    except (IndexError, KeyError) as e:
        if verbose:
            logging.warning(f"Could not match: {rxnorm}, error: {e}")
        atc4_code = None

    return atc4_code


def name_to_atc(
    name: str,
    quickumls_matcher: QuickUMLS,
    umls_cuis_to_rxnorm: Dict[str, str],
    rxnorm_to_atc: Dict[str, str],
    verbose: bool = False,
):
    """
    Convert a drug name/description to an
    ATC code by matching to UMLS, retrieving
    RxNorm concepts and using an API to convert
    RxNorm IDs to ATC

    Parameter
    ---------
    name: str
        the string to map to ATC
    quickumls_matcher: QuickUMLS
        a quickumls matcher object
    umls_cuis_to_rxnorm: Dict[str, str]
        a dict mapping from UMLS CUIs to RxNorm CUIs
    verbose: bool
        -
    """
    if verbose:
        logging.info(f"[eICU] converting: {name}")

    # match the name against UMLS
    matches = quickumls_matcher.match(name, best_match=True, ignore_syntax=False)

    # extract all matches linking to an RxNorm
    atc_matches = []
    for span in matches:
        for match in span:

            umls_cui = match["cui"]
            similarity = match["similarity"]

            if not (umls_cui in umls_cuis_to_rxnorm.keys()):
                continue

            rxnorm_cui = umls_cuis_to_rxnorm[umls_cui]
            if not (rxnorm_cui in rxnorm_to_atc.keys()):
                continue

            atc_code = rxnorm_to_atc[rxnorm_cui]
            atc_matches.append((atc_code, similarity))

    # get closest match to an RxNorm entry
    atc_matches = sorted(atc_matches, key=lambda x: x[1], reverse=True)

    try:
        atc4_code = atc_matches[0][0]

    except (IndexError, KeyError) as e:
        if verbose:
            logging.warning(f"Could not match: {name}, error: {e}")
        atc4_code = None  # type: ignore

    if verbose:
        logging.info(f"[eICU] matched to: {atc4_code}")
    return atc4_code


def name_to_rxnorm_candidates(
    name: str,
    quickumls_matcher: QuickUMLS,
    umls_cuis_to_rxnorm: Dict[str, str],
    verbose: bool = False,
):
    """
    Convert a drug name/description to a
    set of potential RxNorm matches

    Parameter
    ---------
    name: str
        the string to map to ATC
    quickumls_matcher: QuickUMLS
        a quickumls matcher object
    umls_cuis_to_rxnorm: Dict[str, str]
        a dict mapping from UMLS CUIs to RxNorm CUIs
    verbose: bool
        -
    """
    if verbose:
        logging.info(f"[eICU] converting: {name}")

    # match the name against UMLS
    matches = quickumls_matcher.match(name, best_match=True, ignore_syntax=False)

    # extract all matches linking to an RxNorm
    rxnorm_matches = set()
    for span in matches:
        for match in span:

            umls_cui = match["cui"]
            if umls_cui in umls_cuis_to_rxnorm.keys():
                rxnorm_matches.add(umls_cuis_to_rxnorm[umls_cui])

    return rxnorm_matches


def name_chunks_to_rxnorm_candidates(
    names: List[str],
    quickumls_settings: QUSettings,
    umls_cuis_to_rxnorm: Dict[str, str],
    verbose: bool = False,
) -> Set[str]:
    """
    Convert drug name/description to a set of
    potential RxNorm matches
    """

    matcher = QuickUMLS(
        quickumls_settings.quickumls_path,
        overlapping_criteria="score",
        threshold=quickumls_settings.threshold,
        similarity_name=quickumls_settings.similarity,
        window=quickumls_settings.window,
        accepted_semtypes=QUICKUMLS_ACCEPTED_SEMTYPES,
    )

    mapper = partial(
        name_to_rxnorm_candidates,
        quickumls_matcher=matcher,
        umls_cuis_to_rxnorm=umls_cuis_to_rxnorm,
        verbose=False,
    )

    rxnorm_candidate_list = map(mapper, names)
    rxnorm_candidates: Set[str] = set().union(*rxnorm_candidate_list)  # type: ignore

    return rxnorm_candidates


def load_medication_table_eICU(medications_filepath: str, patient_filepath: str) -> pd.DataFrame:
    """
    Load the medication table from eICU

    Parameter
    ---------
    medications_filepath: str
        path to diagnosis.h5 from eICU
    patient_filepath: str
        path to patient.h5 from eICU


    Returns
    -------
    medications_pd: pd.DataFrame
        DF with essential columns:
        ('patienthealthsystemstayid', 'patientunitstayid', 'drugname')
    """
    logging.info(f"[ATC] loading eICU medications table")

    keep_columns = [
        "medicationid",
        "patientunitstayid",
        "drugname",
    ]  # gtc', 'drugname', 'drughiclseqno'
    medications_pd = pd.read_hdf(medications_filepath, columns=keep_columns)

    medications_pd = medications_pd.set_index(keys="medicationid")
    logging.info(f"[ATC] Table raw shape: {medications_pd.shape}")

    medications_pd.dropna(inplace=True)  # drop rows with Nan values
    medications_pd.drop_duplicates(inplace=True)  # drop duplicates

    logging.info(f"[ATC] Load and merge patient table")
    patient_table = pd.read_hdf(
        patient_filepath, columns=["patientunitstayid", "patienthealthsystemstayid"]
    )
    medications_pd = medications_pd.merge(patient_table, on="patientunitstayid")

    return medications_pd


def extract_eICU_medication_mappings(
    medications_filepath: str,
    patient_filepath: str,
    quickumls_path: str,
    umls_mrconso_path: str,
    workers: int = 2,
) -> Tuple[Dict, Dict]:
    """
    Extracts two mappings relevant to the eICU
    medications table
    1) UMLS -> RxNorm
    2) RxNorm -> ATC


    Parameter
    ---------
    medications_filepath: str
        path to diagnosis.h5 from eICU
    patient_filepath: str
        path to patient.h5 from eICU
    quickumls_path: str
        path to quickumls installation
    umls_mrconso_path: str
        path to quickumls mrconso

    Returns
    -------
    umls_cuis_to_rxnorm: Dict
        maps umls cuis to rxnorm ids
    rxnorm_to_atc: Dict
        maps rxnorm ids to ATC codes
    """
    medications_pd = load_medication_table_eICU(medications_filepath, patient_filepath)

    logging.info(f"[RxNorm] load QuickUMLS matcher")
    quickumls_settings = QUSettings(
        quickumls_path=quickumls_path, threshold=0.7, similarity="jaccard", window=8
    )

    logging.info(f"[RxNorm] load mrconso")
    mrconso_df = load_mrconso(umls_mrconso_path, memory_map=True)

    logging.info(f"[RxNorm] compute mapping dict: umls->rxnorm")
    rxnorm_conso_df = mrconso_df[mrconso_df["SAB"] == "RXNORM"]
    umls_cuis_to_rxnorm = {}
    for _, row in tqdm(rxnorm_conso_df.iterrows(), total=len(rxnorm_conso_df)):
        umls_cuis_to_rxnorm[row["CUI"]] = row["CODE"]

    logging.info(f"[RxNorm] compute RxNorm candidate set")
    mapper = partial(
        name_chunks_to_rxnorm_candidates,
        quickumls_settings=quickumls_settings,
        umls_cuis_to_rxnorm=umls_cuis_to_rxnorm,
        verbose=False,
    )

    drugname_list = medications_pd["drugname"].tolist()

    chunk_size = (len(drugname_list) // (64 * workers)) + 1
    list_chunked = [
        drugname_list[i : i + chunk_size] for i in range(0, len(drugname_list), chunk_size)
    ]
    logging.info(
        f"[RxNorm] processing {len(list_chunked)} chunks of {list(map(len, list_chunked))}"
    )

    with mp.Pool(processes=workers) as pool:
        rxnorm_candidate_list = list(tqdm(pool.imap(mapper, list_chunked), total=len(list_chunked)))

    rxnorm_candidates = set().union(*rxnorm_candidate_list)  # type: ignore
    logging.info(f"[RxNorm] found {len(rxnorm_candidates)} candidate ids")

    logging.info(f"[RxNorm] compute mapping to ATC")
    rxnorm_to_atc_map = map(lambda cui: (cui, rxnorm_to_atc_api(cui)), rxnorm_candidates)
    rxnorm_to_atc = {rxnorm_cui: atc for rxnorm_cui, atc in rxnorm_to_atc_map if atc is not None}
    logging.info(f"[RxNorm] resolved mapping to ATC for {len(rxnorm_to_atc)} ids")

    return umls_cuis_to_rxnorm, rxnorm_to_atc


def process_medication_table_eICU(
    medications_filepath: str,
    patient_filepath: str,
    quickumls_path: str,
    umls_cuis_to_rxnorm: Dict,
    rxnorm_to_atc: Dict,
) -> pd.DataFrame:
    """
    Process medication table from eICU

    Parameter
    ---------
    medications_filepath: str
        path to diagnosis.h5 from eICU
    patient_filepath: str
        path to patient.h5 from eICU
    quickumls_path: str
        path to quickumls installation
    umls_cuis_to_rxnorm: Dict
        mapping from UMLS CUIs to RxNorm
    rxnorm_to_atc: Dict
        mapping from RxNorm to ATC codes

    Returns
    -------
    medications_pd: pd.DataFrame
        DF with ATC4 codes and visit, subject identifiers
    """
    medications_pd = load_medication_table_eICU(medications_filepath, patient_filepath)

    # map names to ATC4 codes
    logging.info(f"[ATC] load QuickUMLS matcher")
    quickumls_settings = QUSettings(
        quickumls_path=quickumls_path, threshold=0.7, similarity="jaccard", window=8
    )

    matcher = QuickUMLS(
        quickumls_settings.quickumls_path,
        overlapping_criteria="score",
        threshold=quickumls_settings.threshold,
        similarity_name=quickumls_settings.similarity,
        window=quickumls_settings.window,
        accepted_semtypes=QUICKUMLS_ACCEPTED_SEMTYPES,
    )

    tqdm.pandas()  # tqdm progress for mapping operation
    mapper = partial(
        name_to_atc,
        quickumls_matcher=matcher,
        umls_cuis_to_rxnorm=umls_cuis_to_rxnorm,
        rxnorm_to_atc=rxnorm_to_atc,
        verbose=False,
    )
    medications_pd["ATC4"] = medications_pd["drugname"].progress_map(mapper)
    logging.info(f"[ATC] Table matched shape: {medications_pd.shape}")

    # drop unneeded columns
    medications_pd.drop(
        columns=["drugname"], inplace=True  # ["medicationid", "drugname", "drughiclseqno", "gtc"]
    )

    # rename to match to MIMIC-III naming
    # we treat each hospital stay as an individual "patient"
    # and each ICU stay as a "visit" to perform time-series
    # modeling
    rename_dict = {
        "patientunitstayid": "HADM_ID",  # unit stay -> 'visit'
        "patienthealthsystemstayid": "SUBJECT_ID",  # visit -> 'patient'
    }
    medications_pd = medications_pd.rename(columns=rename_dict)

    logging.info(f"[ATC] Dropping Nan rows")
    medications_pd.dropna(inplace=True)

    logging.info(f"[ATC] Table processed shape: {medications_pd.shape}")
    return medications_pd
