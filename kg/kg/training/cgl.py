# ===============================================
#
# Training Utilities for running disease
# tasks and comparing to CGL, Chet, Sherbet
#
# ===============================================
import argparse
import copy
import datetime
import logging
import pathlib
import random
from os import path
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np

# Tools
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns

# Torch
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchinfo import summary
from tqdm import tqdm, trange

from kg.data.datasets import CglDatasetDownstream, CodeTokenizer
from kg.data.processing import read_set_codes
from kg.model.cgl_modules import CglDownstreamModel, CglDownstreamModel_Lightning
from kg.model.embedding_modules import VocabularyEmbedding
from kg.model.graph_modules import HeterogenousOntologyEmbedding
from kg.model.pretraining_modules import TransformerEncoder
from kg.utils.metrics import metric_report


def train_cgl_downstream(
    graph_embedding: Union[HeterogenousOntologyEmbedding, nn.Embedding],
    encoder: TransformerEncoder,
    tokenizer: CodeTokenizer,
    diagnosis_file: str,
    medications_file: str,
    admissions_file: str,
    data_folder: str,
    patient_ids: Tuple,
    diagnosis_codes: Union[Set[str], Sequence[str]],
    prescription_codes: Union[Set[str], Sequence[str]],
    logger,
    max_sequence_length: int = 64,
    max_visit_length: int = 48,
    batch_size: int = 32,
    num_workers: int = 4,
    embedding_dim: int = 128,
    temporal_attention_dim: int = -1,
    mlp_hidden_dim: int = 128,
    mlp_num_hidden_layers: int = 2,
    num_temporal_contexts: int = 1,
    epochs: int = 1,
    validation_interval: int = 1,
    pretrained: bool = False,
    task: str = "heart_failure",
    learning_rate: float = None,
    umls_graph_data: Dict = None,
    notes_concepts_path: str = None,
    max_visit_text_length: int = 0,
    gradient_accumulation: int = 1,
    es_patience: int = 50,
    store_model_path: str = None,
    cache_graph_forward: bool = False,
    freeze_embedding_after: int = -1,
    eicu_dataset: bool = False,
    down_icd_codes: Optional[str] = None
) -> Tuple[Dict[str, float], ...]:
    """
    Train a model on a disease downstream task and report performance
    """
    with_text = notes_concepts_path is not None

    logging.info("")
    logging.info(40 * "=")
    logging.info(f"Performing downstream DISEASE training")
    logging.info(f"Task: {task}")
    logging.info(f"With text: {with_text}")
    logging.info(f"eICU: {eicu_dataset}")
    logging.info(40 * "=")

    if down_icd_codes is not None:
        down_diagnosis_codes = read_set_codes(down_icd_codes)
        logging.info(f"[CGL] Parsed {len(down_diagnosis_codes)} diagnosis codes from provided file")
    else:
        down_diagnosis_codes = None  # type: ignore

    # unsused branch in this implementation
    if False:
        pass

    else:
        train_ids, val_ids, test_ids = patient_ids
        logging.info(f"[DATA]----- TRAIN -----")
        train_dataset = CglDatasetDownstream(
            diagnosis_file,
            medications_file,
            admissions_file,
            data_folder,
            diagnosis_codes=diagnosis_codes,
            diagnosis_codes_target=down_diagnosis_codes,
            prescription_codes=prescription_codes,
            tokenizer=tokenizer,
            patient_ids=train_ids,
            code_count_range=(1, np.inf),
            visit_range=(1, np.inf),
            max_sequence_length=max_sequence_length,
            patient_sequence_length=max_visit_length,
            target_task=task,
            umls_graph_data=umls_graph_data,
            notes_concepts_path=notes_concepts_path,
            max_visit_text_length=max_visit_text_length,
            eicu=eicu_dataset,
        )

        logging.info(f"[DATA]----- VAL -----")
        val_dataset = CglDatasetDownstream(
            diagnosis_file,
            medications_file,
            admissions_file,
            data_folder,
            diagnosis_codes=diagnosis_codes,
            diagnosis_codes_target=down_diagnosis_codes,
            prescription_codes=prescription_codes,
            tokenizer=tokenizer,
            target_tokenizer=train_dataset.target_tokenizer,
            patient_ids=val_ids,
            code_count_range=(1, np.inf),
            visit_range=(1, np.inf),
            max_sequence_length=max_sequence_length,
            patient_sequence_length=max_visit_length,
            target_task=task,
            umls_graph_data=umls_graph_data,
            notes_concepts_path=notes_concepts_path,
            max_visit_text_length=max_visit_text_length,
            eicu=eicu_dataset,
        )

        logging.info(f"[DATA]----- TEST -----")
        test_dataset = CglDatasetDownstream(
            diagnosis_file,
            medications_file,
            admissions_file,
            data_folder,
            diagnosis_codes=diagnosis_codes,
            diagnosis_codes_target=down_diagnosis_codes,
            prescription_codes=prescription_codes,
            tokenizer=tokenizer,
            target_tokenizer=train_dataset.target_tokenizer,
            patient_ids=test_ids,
            code_count_range=(1, np.inf),
            visit_range=(1, np.inf),
            max_sequence_length=max_sequence_length,
            patient_sequence_length=max_visit_length,
            target_task=task,
            umls_graph_data=umls_graph_data,
            notes_concepts_path=notes_concepts_path,
            max_visit_text_length=max_visit_text_length,
            eicu=eicu_dataset,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    num_target_classes = (
        2
        if task == "heart_failure"
        else len(train_dataset.target_tokenizer.disease_vocabulary.idx2word.values())  # type: ignore
    )

    cgl_model = CglDownstreamModel(
        graph_embedding,
        encoder,
        train_dataset.tokenizer.vocabulary.word2idx["[PAD]"],  # type: ignore
        train_dataset.tokenizer.vocabulary.word2idx["[CLS]"],  # type: ignore
        num_target_classes,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=0.0,
        temporal_attention_dim=temporal_attention_dim,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        num_contexts=num_temporal_contexts,
        with_text=with_text,
        cache_graph_forward=cache_graph_forward,
    )

    # set learning rate
    if learning_rate is None:
        lr = 2e-4 if pretrained else 5e-4
    else:
        lr = learning_rate

    # get training model wrapper
    model_pl = CglDownstreamModel_Lightning(
        cgl_model,
        learning_rate=lr,
        task=task,
        with_text=with_text,
        freeze_embedding_after=freeze_embedding_after,
    )
    # summary(model_pl)

    logging.info(40 * "=")
    logging.info(f"Run Training for {epochs} epochs")
    logging.info(f"Num batches: {len(train_dataloader)}, size: {batch_size}")
    logging.info(f"Val batches: {len(val_dataloader)}")
    logging.info(f"Learning rate: {lr}")
    logging.info(40 * "=")

    # EarlyStopping
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=es_patience)
    logging.info(f"[CGL] downstream training patience: {es_patience}")

    # Checkpointing
    checkpointing = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=2, save_weights_only=True
    )

    if gradient_accumulation > 1:
        accumulated_batch_size = gradient_accumulation * batch_size
        logging.info(f"[TRAIN] gradient accumulation: {gradient_accumulation}")
        logging.info(f"\t-> accumulated batch size: {accumulated_batch_size}")

    # define PL Trainer
    cuda_available = torch.cuda.is_available()
    device_string = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    trainer_pl = pl.Trainer(
        gpus=0 if "cpu" in device_string else 1,
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        check_val_every_n_epoch=validation_interval,
        callbacks=[early_stopping, checkpointing],
        accumulate_grad_batches=gradient_accumulation
    )

    # Train the model ⚡
    trainer_pl.fit(model_pl, train_dataloader, val_dataloader)

    # Load best module
    logging.info(f"Load best validation loss model: {checkpointing.best_model_path}")
    best_model_state_dict = torch.load(checkpointing.best_model_path)["state_dict"]
    missing_keys, unexpected_keys = model_pl.load_state_dict(best_model_state_dict)
    logging.info(f"Missing keys: {missing_keys}")
    logging.info(f"Unexpected keys: {unexpected_keys}")

    if store_model_path is not None:
        logging.info(f"Store cgl model to {store_model_path}")

        pathlib.Path(store_model_path).mkdir(parents=True, exist_ok=True)
        torch.save(model_pl.cgl_model, path.join(store_model_path, f"cgl_{task}_model.pth"))

    # final evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate_model(dataloader):

        collect_probs = []
        collect_targets = []
        model_pl.eval()
        model_pl.to(device)
        with torch.no_grad():

            for batch in tqdm(dataloader):
                batch = tuple(t.to(device) for t in batch)

                if with_text:
                    tokens_d, tokens_p, tokens_text, targets = batch
                    logits, targets = model_pl(
                        tokens_d, tokens_p, targets, x_text=tokens_text, validation=True
                    )
                else:
                    tokens_d, tokens_p, targets = batch
                    logits, targets = model_pl(tokens_d, tokens_p, targets, validation=True)

                if task == "heart_failure":
                    probs = torch.softmax(logits, dim=-1)[:, -1]
                    targets = targets.squeeze(-1)
                else:
                    probs = torch.sigmoid(logits)

                # collect logits, targets
                collect_probs.append(probs)
                collect_targets.append(targets)

            collect_probs = torch.cat(collect_probs).detach().cpu().numpy()
            collect_targets = torch.cat(collect_targets).detach().cpu().numpy()

        # small threshold grid search
        thresholds = np.concatenate((np.linspace(0.01, 0.2, 20), np.linspace(0.2, 0.8, 13)))

        results = []
        for t in thresholds:
            acc_c = model_pl.compute_metrics(np.copy(collect_probs), collect_targets, t=t)
            acc_c["threshold"] = t
            results.append(acc_c)

        best_f1 = sorted(results, key=lambda x: x["f1"], reverse=True)[0]
        logging.info(
            f"Best F1 {best_f1['f1']:.3f}, AuROC: {best_f1['auroc']:.3f}, t: {best_f1['threshold']:.2f}"
        )

        logging.info("----- Sample prediction -----")
        sample_index = random.randint(0, len(collect_probs) - 1)
        logit = collect_probs[sample_index]
        target = collect_targets[sample_index]
        logit = np.where(logit > best_f1["threshold"], 1, 0)

        if task == "heart_failure":
            logging.info(f"Predict: {logit}")
            logging.info(f"Target : {target}")

        else:
            logit_idx = np.where(logit == 1)[0].tolist()
            target_idx = np.where(target == 1)[0].tolist()
            vocabulary = dataloader.dataset.target_tokenizer.disease_vocabulary
            logging.info(
                f"Predict: {' '.join(sorted([vocabulary.idx2word[c] for c in logit_idx]))}"
            )
            logging.info(
                f"Target : {' '.join(sorted([vocabulary.idx2word[c] for c in target_idx]))}"
            )
        logging.info("-----------------------------")

        return best_f1

    logging.info("Running final validation")
    logging.info("Validation set:")
    val_acc = evaluate_model(val_dataloader)
    logging.info("Test set:")
    test_acc = evaluate_model(test_dataloader)

    logging.info(40 * "=")
    logging.info(f"Finished Downstream Training")
    logging.info(f"Val {'F1':<8}: {val_acc['f1']:.3f}")
    if task != "heart_failure":
        logging.info(f"Val {'F1 (inflated)':<8}: {val_acc['f1_inflated']:.3f}")
    logging.info(f"Val {'AuPRC':<8}: {val_acc['auroc']:.3f}")
    logging.info(f"Val {'Thresh':<8}: {val_acc['threshold']:.2f}")
    if task != "heart_failure":
        logging.info(f"Val {'R@20':<8}: {val_acc['recall@20']:.2f}")
    if task != "heart_failure":
        logging.info(f"Val {'R@40':<8}: {val_acc['recall@40']:.2f}")
    logging.info("")
    logging.info(f"Test {'F1':<8}: {test_acc['f1']:.3f}")
    if task != "heart_failure":
        logging.info(f"Test {'F1 (inflated)':<8}: {test_acc['f1_inflated']:.3f}")
    logging.info(f"Test {'AuPRC':<8}: {test_acc['auroc']:.3f}")
    logging.info(f"Test {'Thresh':<8}: {test_acc['threshold']:.2f}")
    if task != "heart_failure":
        logging.info(f"Test {'R@20':<8}: {test_acc['recall@20']:.2f}")
    if task != "heart_failure":
        logging.info(f"Test {'R@40':<8}: {test_acc['recall@40']:.2f}")
    logging.info(40 * "=")

    return val_acc, test_acc
