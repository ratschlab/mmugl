# ===============================================
#
# Data Analysis and Visualization
#
# ===============================================
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch import gru_cell


def diagnosis_statistics(data_pd: pd.DataFrame, percentile: float = 99.5, code: str = "ICD9_CODE"):
    """
    Computes a set of statistics and creates a set of plots
    over a processed MIMIC-III `DIAGNOSES_ICD.csv` or `PRESCRIPTIONS.csv` table

    Parameter
    ---------
    data_pd: processed diagnosis DataFrame
        expected to contain cols: `SUBJECT_ID`, `HADM_ID`, `code` (nested)
    percentile: percentile of codes
    code: column to compute statistics over
    """

    # create subplots
    f, ax = plt.subplots(2, 2, figsize=(14, 8))
    f.subplots_adjust(wspace=0.2, hspace=0.4)

    # global statistics
    patient_count = len(data_pd["SUBJECT_ID"].unique())
    clinical_events = len(data_pd["HADM_ID"].unique())
    logging.info(f"#Patients: {patient_count} and {clinical_events} clinical events")

    # Plot events per patient distribution
    temp = (
        data_pd[["SUBJECT_ID", "HADM_ID"]]
        .groupby(by="SUBJECT_ID")["HADM_ID"]
        .unique()
        .reset_index()
    )
    temp["HADM_ID_Len"] = temp["HADM_ID"].map(lambda x: len(x))
    ax[0, 0].set(yscale="log")
    ax[0, 0].set_title("Distribution of visits per patient")
    ax[0, 0].set_xlabel("Visits per Patient")
    sns.histplot(data=temp, x="HADM_ID_Len", ax=ax[0, 0], bins=40)
    logging.info(f"Min number of events per patient: {temp['HADM_ID_Len'].min()}")
    logging.info(f"Max number of events per patient: {temp['HADM_ID_Len'].max()}")

    diagnosis_codes = data_pd[f"{code}"].values
    unique_diagnosis_codes = set([j for i in diagnosis_codes for j in list(i)])
    logging.info(f"#`{code}` codes (unique): {len(unique_diagnosis_codes)}")

    # Plot distribution of codes per event
    data_pd[f"{code}_Len"] = data_pd[f"{code}"].map(lambda x: len(x))
    ax[0, 1].set_title(f"Distribution {code} codes per event")
    ax[0, 1].set_xlabel("ICD codes per visit")
    sns.histplot(data=data_pd, x=f"{code}_Len", ax=ax[0, 1], bins=40)
    logging.info(f"Min number of codes ({code}) per visit: {data_pd[f'{code}_Len'].min()}")
    logging.info(f"Avg number of codes ({code}) per visit: {data_pd[f'{code}_Len'].mean()}")
    logging.info(f"Median number of codes ({code}) per visit: {data_pd[f'{code}_Len'].median()}")
    logging.info(f"Max number of codes ({code}) per visit: {data_pd[f'{code}_Len'].max()}")

    # Plot distribution of code occurence in visits
    flattened_codes = pd.DataFrame(
        [(index, value) for (index, values) in data_pd[f"{code}"].iteritems() for value in values],
        columns=["index", f"{code}"],
    ).set_index("index")
    grouped_codes = (
        flattened_codes.groupby(by=f"{code}")
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values(by=["count"], ascending=False)
        .reset_index(drop=True)
    )
    total_occurence_count = grouped_codes["count"].sum()
    occurence_percentile = int((percentile / 100) * total_occurence_count)
    count_cumsum = grouped_codes["count"].cumsum()
    code_percentile = count_cumsum[count_cumsum.gt(occurence_percentile)].index[0]
    # code_percentile = np.percentile(grouped_codes['count'], percentile)
    logging.info(f"The {percentile} percentile of codes is at {int(code_percentile)}")

    ax[1, 0].set(xscale="log")
    ax[1, 0].set_title(f"Distribution of code ({code}) occurence in visits")
    ax[1, 0].set_xlabel("Code Occurence")
    sns.histplot(data=grouped_codes, x="count", ax=ax[1, 0], bins=100)
    logging.info(f"Max number of occurence of a code: {grouped_codes['count'].max()}")
