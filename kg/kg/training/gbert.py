# ===============================================
#
# Training Utilities for running Gbert
#
# Some code inspired by: https://github.com/jshang123/G-Bert
# ===============================================
import argparse
import datetime
import logging
import random
from os import path
from typing import Any, Dict, Optional, Tuple

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

from kg.data.cached_datasets import GbertDatasetDownstream
from kg.data.datasets import CodeTokenizer
from kg.model.embedding_modules import VocabularyEmbedding
from kg.model.gbert_modules import GbertDownstreamModel, GbertDownstreamModel_Lightning
from kg.model.pretraining_modules import TransformerEncoder, VisitLatentSpaceProjector
from kg.utils.metrics import metric_report


def train_gbert_downstream(
    graph_embedding: VocabularyEmbedding,
    encoder: TransformerEncoder,
    tokenizer: CodeTokenizer,
    gbert_data: str,
    logger,
    max_sequence_length: int = 47,
    max_visit_length: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    embedding_dim: int = 128,
    epochs: int = 1,
    validation_interval: int = 1,
    admissions_file_path: Optional[str] = None,
    pretrained: bool = False,
    latent_space_projector: VisitLatentSpaceProjector = None,
    gated_graph_lookup: bool = False,
    attention_heads: int = 1,
    umls_graph_data: Dict = None,
    notes_concepts_path: str = None,
    max_visit_text_length: int = 0,
    gradient_accumulation: int = 1,
    es_patience: int = 12,
    visit_model: str = "avg",
    learning_rate: float = None,
    mlp_hidden_dim: int = 128,
    mlp_num_hidden_layers: int = 1,
) -> Tuple[Dict[str, float], ...]:
    """
    Train a model on the Gbert downstream task and report performance
    """
    with_text = notes_concepts_path is not None

    logging.info("")
    logging.info(40 * "=")
    logging.info(f"Performing downstream Gbert training")
    logging.info(f"With text: {with_text}")
    logging.info(40 * "=")

    train_dataset = GbertDatasetDownstream(
        data_dir=gbert_data,
        split="train",
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        patient_sequence_length=max_visit_length,
        admissions_file_path=admissions_file_path,
        umls_graph_data=umls_graph_data,
        notes_concepts_path=notes_concepts_path,
        max_visit_text_length=max_visit_text_length,
    )

    val_dataset = GbertDatasetDownstream(
        data_dir=gbert_data,
        split="val",
        tokenizer=tokenizer,
        target_tokenizer=train_dataset.target_tokenizer,
        max_sequence_length=max_sequence_length,
        patient_sequence_length=max_visit_length,
        admissions_file_path=admissions_file_path,
        umls_graph_data=umls_graph_data,
        notes_concepts_path=notes_concepts_path,
        max_visit_text_length=max_visit_text_length,
    )

    test_dataset = GbertDatasetDownstream(
        data_dir=gbert_data,
        split="test",
        tokenizer=tokenizer,
        target_tokenizer=train_dataset.target_tokenizer,
        max_sequence_length=max_sequence_length,
        patient_sequence_length=max_visit_length,
        admissions_file_path=admissions_file_path,
        umls_graph_data=umls_graph_data,
        notes_concepts_path=notes_concepts_path,
        max_visit_text_length=max_visit_text_length,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    num_target_classes = len(
        train_dataset.target_tokenizer.prescription_vocabulary.idx2word.values()
    )

    # gated graph lookup setup
    if gated_graph_lookup:
        gated_graph_lookup_nodes = ["disease", "prescription"]
    else:
        gated_graph_lookup_nodes = None  # type: ignore

    gbert_model = GbertDownstreamModel(
        graph_embedding,
        encoder,
        padding_id=train_dataset.tokenizer.vocabulary.word2idx["[PAD]"],
        cls_id=train_dataset.tokenizer.vocabulary.word2idx["[CLS]"],
        num_classes=num_target_classes,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=0.0,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        latent_space_projector=latent_space_projector,
        gated_graph_lookup=gated_graph_lookup_nodes,
        attention_heads=attention_heads,
        with_text=with_text,
        visit_model=visit_model,
    )

    # set learning rate
    if learning_rate is None:
        lr = 1e-4 if pretrained else 5e-4
    else:
        lr = learning_rate

    model_pl = GbertDownstreamModel_Lightning(gbert_model, learning_rate=lr, with_text=with_text)
    # summary(model_pl)

    logging.info(40 * "=")
    logging.info(f"Run Training for {epochs} epochs")
    logging.info(f"Num batches: {len(train_dataloader)}, size: {batch_size}")
    logging.info(f"Val batches: {len(val_dataloader)}")
    logging.info(f"Learning rate: {lr}")
    logging.info(40 * "=")

    # EarlyStopping
    early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=es_patience)
    logging.info(f"[TRAIN] downstream training patience: {es_patience}")

    # Checkpointing
    checkpointing = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=2, save_weights_only=True
    )

    if gradient_accumulation > 1:
        accumulated_batch_size = gradient_accumulation * batch_size
        logging.info(f"[TRAIN] gradient accumulation: {gradient_accumulation}")
        logging.info(f"\t-> accumulated batch size: {accumulated_batch_size}")

    # define PL Trainer
    trainer_pl = pl.Trainer(
        gpus=1,
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

    # final evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate_model(dataloader):

        collect_logits = []
        collect_targets = []
        model_pl.eval()
        model_pl.to(device)
        with torch.no_grad():

            for batch in tqdm(dataloader):

                batch = tuple(t.to(device) for t in batch)
                if with_text:
                    tokens_d, tokens_p, tokens_text, targets_p = batch
                    _, logits, targets, target_mask = model_pl(
                        tokens_d, tokens_p, targets_p, x_text=tokens_text
                    )
                else:
                    tokens_d, tokens_p, targets_p = batch
                    _, logits, targets, target_mask = model_pl(tokens_d, tokens_p, targets_p)
                logits = torch.sigmoid(logits)

                # collect unmasked logits, targets
                for logit, target, mask in zip(logits, targets, target_mask):
                    if mask == 1:
                        collect_logits.append(logit)
                        collect_targets.append(target)

            collect_logits = torch.stack(collect_logits).detach().cpu().numpy()
            collect_targets = torch.stack(collect_targets).detach().cpu().numpy()

        # small threshold grid search
        thresholds = np.linspace(0.1, 0.9, 9)
        results = []
        for t in thresholds:
            acc_c = metric_report(
                np.copy(collect_logits), collect_targets, threshold=t, verbose=False
            )
            acc_c["threshold"] = t
            results.append(acc_c)

        best_jaccard = sorted(results, key=lambda x: x["jaccard"], reverse=True)[0]
        best_f1 = sorted(results, key=lambda x: x["f1"], reverse=True)[0]
        logging.info(
            f"Best Jaccard {best_jaccard['jaccard']:.3f}, at F1: {best_jaccard['f1']:.3f}, t: {best_jaccard['threshold']:.2f}"
        )
        logging.info(
            f"Best F1 {best_f1['f1']:.3f}, at Jaccard: {best_f1['jaccard']:.3f}, t: {best_f1['threshold']:.2f}"
        )

        logging.info("----- Sample prediction -----")
        sample_index = random.randint(0, len(collect_logits) - 1)
        logit = collect_logits[sample_index]
        target = collect_targets[sample_index]
        logit = np.where(logit > best_jaccard["threshold"], 1, 0)

        logit_idx = np.where(logit == 1)[0].tolist()
        target_idx = np.where(target == 1)[0].tolist()
        vocabulary = dataloader.dataset.target_tokenizer.prescription_vocabulary
        logging.info(f"Predict: {' '.join(sorted([vocabulary.idx2word[c] for c in logit_idx]))}")
        logging.info(f"Target : {' '.join(sorted([vocabulary.idx2word[c] for c in target_idx]))}")
        logging.info("-----------------------------")

        return best_jaccard

    logging.info("Running final validation")
    logging.info("Validation set:")
    val_acc = evaluate_model(val_dataloader)
    logging.info("Test set:")
    test_acc = evaluate_model(test_dataloader)

    logging.info(40 * "=")
    logging.info(f"Finished Downstream Training")
    logging.info(f"Val {'Jaccard':<8}: {val_acc['jaccard']:.3f}")
    logging.info(f"Val {'F1':<8}: {val_acc['f1']:.3f}")
    logging.info(f"Val {'AuPRC':<8}: {val_acc['prauc']:.3f}")
    logging.info(f"Val {'Thresh':<8}: {val_acc['threshold']:.2f}")
    logging.info("")
    logging.info(f"Test {'Jaccard':<8}: {test_acc['jaccard']:.3f}")
    logging.info(f"Test {'F1':<8}: {test_acc['f1']:.3f}")
    logging.info(f"Test {'AuPRC':<8}: {test_acc['prauc']:.3f}")
    logging.info(f"Test {'Thresh':<8}: {test_acc['threshold']:.2f}")
    logging.info(40 * "=")

    return val_acc, test_acc
