# ===============================================
#
# Utility code around training
#
# ===============================================
import logging
import pickle
import subprocess
from os import path
from typing import Dict, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import torch.nn as nn

from kg.data.processing import process_diagnosis_table, split_dataset


def freeze_layer(layer: nn.Module):
    """Freeze all parameters of a given torch layer"""

    for param in layer.parameters():
        param.requires_grad = False

    logging.info(f"Layer: {layer} frozen, ({len(list(layer.parameters()))} params frozen)")


def load_from_pickle(filepath):
    with open(filepath, "rb") as handle:
        d = pickle.load(handle)
    return d


# ========================
# Patient Splitting
# ========================
def get_splits(
    diagnosis_csv: pd.DataFrame,
    mode: str = "random",
    random_state: int = 42,
    testing: bool = False,
    data_dir: str = None,
) -> Tuple[Optional[Union[Set, Sequence]], ...]:
    """
    Create train/val/test splits in different ways:
    - random: random split over all possible patients
    - precomputed: load from provided file path, keep val, test split
        ids; consider the rest for training

    Parameter
    ---------
    diagnosis_csv: path to diagnosis table
    mode: how to split the patients
    random_state: -
    testing: - 
    data_dir: path to directory with precomputed splits
    """

    # for testing
    if testing:
        return list(range(20)), list(range(20, 40))

    # load candidate ids from diagnosis table
    if "eicu" not in mode:
        diagnosis_pd = process_diagnosis_table(diagnosis_csv)
        test_ids = None
    else:
        logging.info(f"[SPLIT] load cached eICU diagnosis table: {diagnosis_csv}")
        with open(diagnosis_csv, "rb") as handle:
            diagnosis_pd = pickle.load(handle)

    # random mode
    if mode == "random":
        train_ids, val_ids = split_dataset(
            diagnosis_pd, random_state=random_state
        )

    elif mode == "random-test":
        return split_dataset(
            diagnosis_pd,
            train=0.7,
            val=0.15,
            test=True,
            single_visit_train=True,
            random_state=random_state,
        )

    elif mode == "random-test-eicu":
        return split_dataset(
            diagnosis_pd,
            train=0.7,
            val=0.15,
            test=True,
            single_visit_train=True,
            random_state=random_state,
        )

    elif mode == "small":
        logging.warning(f"Testing on small data subset")
        return split_dataset(
            diagnosis_pd[0:1000],
            train=0.7,
            val=0.15,
            test=True,
            single_visit_train=True,
            random_state=random_state,
        )

    elif "precomputed" in mode:
        all_patient_ids = set(diagnosis_pd["SUBJECT_ID"].unique())
        assert data_dir is not None
        precomputed_dir = data_dir

        logging.info(f"[SPLIT] considering {len(all_patient_ids)} patients")

        def extract_patient_ids(dir: str, split: str):
            file_path = path.join(dir, f"{split}_listfile.csv")
            patient_ids = (
                pd.read_csv(file_path)["filename"].apply(lambda x: int(x.split("_")[0])).unique()
            )

            logging.info(
                f"[SPLIT] Extracted {len(patient_ids)} patients for split: `{split}` from file"
            )
            return patient_ids

        suggested_train_ids = set(extract_patient_ids(precomputed_dir, "train"))
        val_ids = set(extract_patient_ids(precomputed_dir, "val"))  # type: ignore
        test_ids = set(extract_patient_ids(precomputed_dir, "test"))

        if mode == "precomputed-plus":
            train_ids = all_patient_ids - (val_ids.union(test_ids))  # type: ignore
        elif mode == "precomputed":
            train_ids = suggested_train_ids  # type: ignore
        else:
            logging.error(f"[SPLIT] Unsupported split mode {mode}")
            exit()

    elif mode in ("cgl", "sherbet", "chet"):

        all_patient_ids = set(diagnosis_pd["SUBJECT_ID"].unique())
        assert data_dir is not None
        split_dir = data_dir
        split_file = f"{mode}_patient_splits.pkl"

        precomputed_splits = load_from_pickle(path.join(split_dir, split_file))

        val_ids = set(precomputed_splits["val"])  # type: ignore
        test_ids = set(precomputed_splits["test"])  # type: ignore

        suggested_train_ids = set(precomputed_splits["train"])
        train_ids = all_patient_ids - (val_ids.union(test_ids))  # type: ignore
        train_intersection = train_ids.intersection(suggested_train_ids)  # type: ignore
        logging.info(f"[SPLIT {mode}] train suggested: {len(suggested_train_ids)}")
        logging.info(f"[SPLIT {mode}] train loaded: {len(train_ids)}")
        logging.info(f"[SPLIT {mode}] intersect: {len(train_intersection)}")

    else:
        logging.error(f"[SPLIT] Unsupported split mode {mode}")
        exit()

    return train_ids, val_ids, test_ids
