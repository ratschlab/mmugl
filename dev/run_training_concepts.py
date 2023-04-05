#!/usr/bin/env python3
# ===============================================
#
# Training Script
#
# ===============================================
import logging
import coloredlogs
import datetime
import argparse
import pathlib
import socket
import pickle
from os import path
from typing import Dict, Any, Set, Tuple, Sequence, Union, Optional
from sklearn.model_selection import validation_curve

# Tools
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# Custom
import kg
from kg.data.processing import process_diagnosis_table, split_dataset, \
        filter_diagnosis_table, get_unique_codes, group_diagnosis_table, \
        write_set_codes, read_set_codes
from kg.data.graph import CoLinkConfig
from kg.data.datasets import CodeDataset
from kg.data.vocabulary import CodeTokenizer
from kg.data.cached_datasets import GbertDataset
from kg.model.pretraining_modules import JointCodePretrainingTransformer
from kg.utils.metrics import metric_report
from kg.utils.training import freeze_layer, get_splits
from kg.model.lightning_modules import JointCodePretrainingTransformer_Lightning
from kg.training.gbert import train_gbert_downstream
from kg.training.cgl import train_cgl_downstream
from kg.data.umls import load_umls_graph


# ========================
# GLOBAL
# ========================
TESTING = False
LOGGING_LEVEL = "INFO"
VALIDATION_INTERVAL = 2
MAX_SEQUENCE_LENGTH = 63

# ========================
# Argparse
# ========================
def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("-v", "--verbose", action="store_true", help="More verbose output")
    parser.add_argument("--logdir", type=str, default="./runs", help="Output log directory")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--downstream_gbert", action='store_true',
        help="Perform downstream Gbert training/evaluation")
    parser.add_argument("--downstream_cgl", action='store_true',
        help="Perform downstream CGL training/evaluation on heart failure")

    # Data related arguments
    parser_data = parser.add_argument_group("Data")
    parser_data.add_argument("--data_dir", required=True, type=str,
        help="Path to main data directory with downloaded data")
    parser_data.add_argument("--diagnosis_csv", required=True, type=str,
        help="Path to MIMIC-III `DIAGNOSES_ICD.csv")
    parser_data.add_argument("--prescription_csv", required=True, type=str,
        help="Path to MIMIC-III `PRESCRIPTIONS.csv")
    parser_data.add_argument("--admission_csv", required=True, type=str,
        help="Path to MIMIC-III `ADMISSIONS.csv")
    parser_data.add_argument("--icd_codes", required=True, type=str,
        help="Path to .txt containing list of considered ICD codes")
    parser_data.add_argument("--down_icd_codes", default=None, type=str,
        required=True, help="Path to .txt containing list of considered ICD codes for downstream targets")
    parser_data.add_argument("--atc_codes", required=True, type=str,
        help="Path to .txt containing list of considered ATC codes")
    parser_data.add_argument("--code_mappings", required=True, type=str,
        help="Path to directory for NDC -> ATC code mappings")
    parser_data.add_argument("--split_mode", default="random-test", type=str,
        help="Mode used for creating train/val/... splits; options: {random, precomputed}")
    parser_data.add_argument("--gbert_data", default=None, type=str,
        help="directory containing gbert repo provided data, if None will not use")
    parser_data.add_argument("--co_occurrence", default=None, type=str,
        help="Wether to also build co-occurrence into the graph based on the training set; pass one of: {visit, patient}")
    parser_data.add_argument("--co_occurrence_subsample", default=0.3, type=float,
        help="Subsampling ratio for co-occurrence nodes, in [0, 1]")
    parser_data.add_argument("--co_occurrence_cluster", type=int, default=0,
        help="reduce num co nodes by clustering")
    parser_data.add_argument("--co_occurrence_features", default=None, type=str,
        help="path to DataFrame with stored static node features")
    parser_data.add_argument("--store_graphmodel", default=False, action="store_true",
        help="Whether to store the learned graph module and the node embeddings")
    parser_data.add_argument("--store_pretrainmodel", default=False, action="store_true",
        help="Whether to store the full model after pretraining")
    parser_data.add_argument("--notes_concepts", default=None, type=str,
        help="Path to pre-extracted concepts for each note")
    parser_data.add_argument("--eicu_dataset", default=False, action="store_true",
        help="Whether this training is on the eICU dataset")

    # Training related arguments
    parser_training = parser.add_argument_group("Training")
    parser_training.add_argument("--name", type=str, default="pretraining_icd", help="Name for training run")
    parser_training.add_argument("--num_workers", type=int,
        default=4, help="Number of torch dataloader workers")
    parser_training.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser_training.add_argument("--down_batch_size", type=int, default=None,
        help="Dowstream training batch size")
    parser_training.add_argument("--epochs", type=int, default=1, help="#Epochs of training")
    parser_training.add_argument("--learning_rate", type=float, default=5e-4, help="Optimizer learning rate")
    parser_training.add_argument("--down_learning_rate", type=float, default=None, help="Optimizer learning rate for ds training")
    parser_training.add_argument("--pretrain", default=False, action="store_true",
        help="Enable pretraining")
    parser_training.add_argument("--freeze_embeddings", default=False, action="store_true",
        help="Freeze the graph or embedding layers after pretraining i.e. during downstream training")
    parser_training.add_argument("--freeze_encoder", default=False, action="store_true",
        help="Freeze the Transformer encoder after pretraining i.e. during downstream training")
    parser_training.add_argument("--pretrain_es", type=str, default="loss",
        help="Pretraining ES target 'val/{}'")
    parser_training.add_argument("--pretrain_patience", type=int, default=6,
        help="Patience for ES during pretraining")
    parser_training.add_argument("--cgl_task", default="heart_failure", type=str,
        help="CGL downstream task: {heart_failure, diagnosis}")
    parser_training.add_argument("--no_cache_graph_forward", default=False, action='store_true')
    parser_training.add_argument("--no_2p_task_loss", default=False, action='store_true',
        help="If given (true) there is no loss signal for predicting prescription in pretraining")
    parser_training.add_argument("--grad_acc", type=int, default=1,
        help="Number of batches to accumulate per gradient step, default 1 means no gradient accumulation")

    # Model related arguments
    parser_model = parser.add_argument_group("Model")
    parser_model.add_argument("--embedding_dim", type=int, default=128,
        help="Dimension for the embeddings, graph layers and transformer")
    parser_model.add_argument("--graph_num_layers", type=int, default=1,
        help="Number of GNN layers")
    parser_model.add_argument("--graph_num_filters", type=int, default=1,
        help="Number of distinct GNN layer stacks")
    parser_model.add_argument("--graph_staged", default=False, action="store_true",
        help="Whether to run the GNN `staged` as in G-Bert")
    parser_model.add_argument("--attention_heads", type=int, default=1, 
        help="Number of attention heads in the transformer")
    parser_model.add_argument("--feedforward_dim", type=int, default=512, 
        help="MLP dimension for the transformer, usually 4*the transformer model dimension")
    parser_model.add_argument("--num_blocks", type=int, default=1,
        help="Number of transformer blocks")
    parser_model.add_argument("--mlp_dim", type=int, default=128,
        help="Dimension for the classifier MLPs")
    parser_model.add_argument("--mlp_num_layers", type=int, default=2,
        help="Number of MLP layers in the classifiers")
    parser_model.add_argument("--agg_mask_loss", type=float, default=0.25,
        help="Weight for the aggregated masked loss; CLS loss weighted with 1.0")
    parser_model.add_argument("--set_loss_alpha", type=float, default=0.0,
        help="Weight for the aggregated masked loss; CLS loss weighted with 1.0")
    parser_model.add_argument("--occurrence_loss_alpha", type=float, default=0.0,
        help="Weight for the co-occurrence encoder loss, only relevant if using co-occurrence nodes")
    parser_model.add_argument("--occurrence_dim_divisor", type=int, default=4,
        help="Divisor to reduce occurrence node dimension to save memory")
    parser_model.add_argument("--co_occurrence_dropout", type=float, default=0.0,
        help="co_occurrence_dropout: dropout to apply on the edges towards the co-nodes")
    parser_model.add_argument("--co_latent_space", default=False, action="store_true",
        help="Use the co-nodes from the graph as latent space for the visit encoding")
    parser_model.add_argument("--decoder_network", default=False, action="store_true",
        help="Whether to use a decoder network, which attends back to all the graph nodes")
    parser_model.add_argument("--gnn_operator", default="GCNConv", type=str,
        help="GNN operator to use {GCNConv, GATConv, ...}")
    parser_model.add_argument("--triplet_loss_alpha", default=0.0, type=float,
        help="Alpha parameter for additional triplet loss on occurrence node embeddings")
    parser_model.add_argument("--triplet_margin", default=0.1, type=float, 
        help="Margin parameter for the triplet loss")
    parser_model.add_argument("--graph_memory_size", default=0, type=int, 
        help="Use a compression memory on the co nodes of the graph")

    # ===== Co Links ======
    parser_model.add_argument("--co_link_type", default=None, type=str,
        help="co occurrence type for the co links: {patient, visit}")
    parser_model.add_argument("--co_link_edge_weight", default=False, action="store_true",
        help="Weighted edges for co links")
    parser_model.add_argument("--co_link_alpha_inter", default=1.0, type=float,
        help="Alpha parameter for co link inter node type")
    parser_model.add_argument("--co_link_alpha_intra", default=1.0, type=float,
        help="Alpha parameter for co link intra node type")
    parser_model.add_argument("--trainable_edge_weights", default=False, action="store_true",
        help="Make edge weights trainable")

    # ===== Contractions ======
    parser_model.add_argument("--contractions_type", default=None, type=str,
        help="Perform graph contractions i.e. edge pooling with provided scoring func")

    # ===== UMLS ======
    parser_model.add_argument("--umls_graph", default=None, type=str,
        help="path to a pre-extracted UMLS CUI graph")
    parser_model.add_argument("--max_visit_text_length", default=0, type=int,
        help="Maximum length of text concept tokens")
    parser_model.add_argument("--text_target_loss", default=0.0, type=float,
        help="Additional target to predict encoded text tokens")
    
    # ===== Downstream ======
    parser_model.add_argument("--gated_lookup", default=False, action="store_true",
        help="Perform a gated lookup onto the graph in downstream training")
    parser_model.add_argument("--down_mlp_dim", type=int, default=None,
        help="Dimension for the classifier MLPs (downstream tasks)")
    parser_model.add_argument("--down_mlp_num_layers", type=int, default=None,
        help="Number of MLP layers in the classifiers (downstream tasks)")
    parser_training.add_argument("--down_grad_acc", type=int, default=1,
        help="Number of batches to accumulate per gradient step, default 1 means no gradient accumulation")
    parser_training.add_argument("--down_es", type=int, default=16,
        help="Early stopping patience for downstream training")
    parser_data.add_argument("--store_downmodel", default=False, action="store_true",
        help="Whether to store the full model after downstream training")
    parser_training.add_argument("--down_freeze_emb_after", type=int, default=-1,
        help="Freeze graph embedding downstream only after x epochs")
    
    # ===== CGL ======
    parser_model.add_argument("--cgl_num_contexts", default=1, type=int, 
        help="Number of attention contexts to use for the temporal attention in the CGL downstream task")

    # ===== GBERT ======
    parser_model.add_argument("--gbert_visit_model", default='avg', type=str, 
        help="Visit model for the Gbert med. recommendation task, default past averaging")

    args = parser.parse_args()
    return args

# ========================
# Print Model Config
# ========================
def print_model_configuration(args: argparse.Namespace):
    logging.info("----- Model -----")
    logging.info(f"Embedding dimension: {args.embedding_dim}")

    if args.graph_num_layers > 0:
        logging.info(f"GNN operator: {args.gnn_operator}")
        logging.info(f"Graph layers: {args.graph_num_layers}, staged: {args.graph_staged}")
    else:
        logging.info(f"Using normal embedding layer, no graph")

    logging.info(
        f"Transformer heads: {args.attention_heads}, mlp: {args.feedforward_dim}, blocks: {args.num_blocks}")
    logging.info(f"MLP dim: {args.mlp_dim}, layers: {args.mlp_num_layers}")
    logging.info(f"Masked loss alpha: {args.agg_mask_loss}")
    logging.info(f"Hungarian set loss alpha: {args.set_loss_alpha}")
    logging.info(f"Co-occurrence loss alpha: {args.occurrence_loss_alpha}")
    logging.info(f"co-nodes latent space: {args.co_latent_space}")
    logging.info(f"Decoder Network: {args.decoder_network}")
    logging.info("----- ----- -----")



# ========================
# MAIN
# ========================
def main():
    """Training Script procedure"""
    if TESTING:
        logging.warning("TESTFLAG SET")

    # Parse CMD arguments
    args = parse_arguments()
    if args.gbert_data is not None and args.gbert_data in {"", "-"}:
        args.gbert_data = path.join(args.data_dir, "gbert")
        logging.info(f"[G-Bert] set data directory to: {args.gbert_data}")

    # get GPU availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_string = torch.cuda.get_device_name(0) if cuda_available else 'cpu'
    logging.info(40*"=")
    logging.info("Start Training script")
    logging.info(f"Host: {socket.gethostname()}")
    logging.info(f"Torch device: {device_string}")
    if cuda_available:
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        logging.info(f"GPU Memory: {gpu_memory} GB")
    logging.info(f"Workers: {args.num_workers}")
    logging.info(40*"=")

    # get hparams
    hparams = vars(args)
    hparams['max_seq_length'] = MAX_SEQUENCE_LENGTH
    
    # set seed
    pl.seed_everything(args.random_state, workers=True)

    # Load datasets
    # unsued branch for this implementation
    if False:
        pass

    else:

        # Load UMLS graph if required
        if args.umls_graph is not None:
            logging.info(f"Load UMLS graph from: {args.umls_graph}")
            umls_graph = load_umls_graph(args.umls_graph, True)
        else:
            umls_graph = None


        # create datasets for training and validation
        if args.gbert_data is None or args.eicu_dataset:

            # Create data splits
            train_ids, val_ids, test_ids = get_splits(args.diagnosis_csv,
                mode=args.split_mode,
                random_state=args.random_state,
                testing=TESTING,
                data_dir=args.data_dir)
            logging.info(f"[SPLIT] Created data splits, train: {len(train_ids)}, val: {len(val_ids)}") 

            # Get codes
            disease_codes = read_set_codes(args.icd_codes)
            logging.info(f"Parsed {len(disease_codes)} diagnosis codes from provided file")
            prescription_codes = read_set_codes(args.atc_codes)
            logging.info(f"Parsed {len(prescription_codes)} prescription codes from provided file")

            logging.info(f"[DATA]----- TRAIN -----")
            train_dataset = CodeDataset(
                args.diagnosis_csv,
                args.prescription_csv,
                args.code_mappings,
                disease_codes,
                prescription_codes,
                patient_ids=train_ids,
                code_count_range=(1, np.inf), # (2, np.inf),
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                random_masking_probability=0.15,
                code_shuffle=True,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length,
                text_targets=args.text_target_loss > 0.0
            )

            logging.info(f"[DATA]----- VAL -----")
            val_dataset = CodeDataset(
                args.diagnosis_csv,
                args.prescription_csv,
                args.code_mappings,
                disease_codes,
                prescription_codes,
                patient_ids=val_ids,
                code_count_range=(1, np.inf), # (2, np.inf),
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                random_masking_probability=0.15,
                tokenizer=train_dataset.tokenizer,
                code_shuffle=True,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length,
                text_targets=args.text_target_loss > 0.0
            )

            
        else: # use G-Bert Datasets

            train_dataset = GbertDataset(
                args.gbert_data,
                split="train",
                pretraining=True,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                random_masking_probability=0.15,
                code_shuffle=True,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length
            )

            val_dataset = GbertDataset(
                args.gbert_data,
                split="val",
                pretraining=True,
                tokenizer=train_dataset.tokenizer,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                random_masking_probability=0.15,
                code_shuffle=True,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length
            )

    # setup dataloaders
    train_dataloader = DataLoader(train_dataset,
        shuffle=True, batch_size = args.batch_size, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=3)

    val_dataloader = DataLoader(val_dataset,
        shuffle=False, batch_size = args.batch_size, num_workers=args.num_workers)

    logger = TensorBoardLogger(
            args.logdir,
            name=f"tb",
            default_hp_metric=False
        )

    # Co Link Config
    if args.co_link_type is not None:
        co_link_config = CoLinkConfig(
            link_type=args.co_link_type,
            edge_weights=args.co_link_edge_weight,
            normalize_weights=True if args.co_link_edge_weight else False,
            alpha_inter=args.co_link_alpha_inter,
            alpha_intra=args.co_link_alpha_intra
        )
    else:
        co_link_config = None

    bce_logit_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
    pretrain_model = JointCodePretrainingTransformer(
        train_dataset.tokenizer,
        train_dataset.tokenizer.vocabulary.word2idx["[PAD]"],
        train_dataset.tokenizer.vocabulary.word2idx["[CLS]"],
        embedding_dim=args.embedding_dim,
        graph_num_layers=args.graph_num_layers,
        graph_num_filters=args.graph_num_filters,
        graph_staged=args.graph_staged,
        mlp_dim=args.mlp_dim,
        mlp_num_layers=args.mlp_num_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        num_blocks=args.num_blocks,
        agg_mask_output=args.agg_mask_loss > 0,
        token_logit_output=args.set_loss_alpha > 0,
        decoder_network=args.decoder_network,
        convolution_operator=args.gnn_operator,
        data_pd=train_dataset.data_pd,
        co_occurrence=args.co_occurrence,
        co_occurrence_subsample=args.co_occurrence_subsample,
        co_occurrence_loss=args.occurrence_loss_alpha,
        co_occurrence_divisor=args.occurrence_dim_divisor,
        co_occurrence_dropout=args.co_occurrence_dropout,
        co_latent_space=args.co_latent_space,
        co_occurrence_cluster=args.co_occurrence_cluster,
        co_occurrence_features=args.co_occurrence_features,
        co_occurrence_batch_size=int(args.batch_size // 2),
        triplet_batch_size=int(args.batch_size // 2),
        triplet_margin=args.triplet_margin,
        triplet_loss=args.triplet_loss_alpha,
        graph_memory_size=args.graph_memory_size,
        umls_graph=umls_graph,
        co_link_config=co_link_config,
        with_text=(args.notes_concepts is not None),
        contractions_type=args.contractions_type,
        trainable_edge_weights=args.trainable_edge_weights,
        cache_graph_forward=(not args.no_cache_graph_forward),
        text_targets=args.text_target_loss > 0.0
    )
    # pretrain_model.to(device)
    if args.verbose: print_model_configuration(args)

    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_name = f"{args.name}_{time_string}"
    log_dir = f"{args.logdir}/{run_name}"
    logger.experiment.add_text("name", run_name, 0)
    logging.info(f"Tensorboard logs at: {log_dir}")
    
    logging.info(40*"=")
    logging.info(f"Run Training for {args.epochs} epochs")
    logging.info(f"Num batches: {len(train_dataloader)}, size: {args.batch_size}")
    logging.info(f"Val batches: {len(val_dataloader)}")
    logging.info(f"Learning rate: {args.learning_rate}")
    logging.info(40*"=")  

    # configure early stopping
    early_stopping_patience = args.pretrain_patience
    logging.info(f"[TRAIN] early stopping on val/{args.pretrain_es}, patience {early_stopping_patience}")
    early_stopping = EarlyStopping(monitor=f"val/{args.pretrain_es}",
        mode="min", patience=early_stopping_patience)

    # Checkpointing
    checkpointing = ModelCheckpoint(
        monitor="val/loss", mode="min", save_top_k=2, save_weights_only=True
    )

    # Pytorch Lightning Training Model
    model_pl = JointCodePretrainingTransformer_Lightning(
        pretrain_model, bce_logit_loss,
        learning_rate=args.learning_rate,
        masked_loss_alpha=args.agg_mask_loss,
        set_loss_alpha=args.set_loss_alpha,
        occurrence_loss_alpha=args.occurrence_loss_alpha,
        triplet_loss_alpha=args.triplet_loss_alpha,
        batch_size=args.batch_size,
        with_text=(args.notes_concepts is not None),
        text_target_loss=args.text_target_loss,
        no_2p_task_loss=args.no_2p_task_loss,
    )

    gradient_accumulation = args.grad_acc
    if gradient_accumulation > 1:
        accumulated_batch_size = gradient_accumulation * args.batch_size
        logging.info(f"[TRAIN] gradient accumulation: {gradient_accumulation}")
        logging.info(f"\t-> accumulated batch size: {accumulated_batch_size}")

    # define PL Trainer
    num_gpus = 0 if TESTING or 'cpu' in device_string else 1
    trainer_pl = pl.Trainer(
        gpus=num_gpus,
        max_epochs=args.epochs,
        logger=logger,
        enable_checkpointing=True, # False,
        check_val_every_n_epoch=VALIDATION_INTERVAL,
        callbacks=[early_stopping, checkpointing],
        accumulate_grad_batches=gradient_accumulation,
    )

    # Train the model ⚡
    if not (args.downstream_gbert and TESTING) and args.pretrain:
        trainer_pl.fit(model_pl, train_dataloader, val_dataloader)

        # Load best module
        logging.info(f"Load best validation loss model: {checkpointing.best_model_path}")
        best_model_state_dict = torch.load(checkpointing.best_model_path)["state_dict"]
        missing_keys, unexpected_keys = model_pl.load_state_dict(best_model_state_dict)
        logging.info(f"Missing keys: {missing_keys}")
        logging.info(f"Unexpected keys: {unexpected_keys}")

    else:
        logging.warning(f"Skipping pretraining due to test flag `{TESTING}` or pretrain `{args.pretrain}`")


    if args.store_pretrainmodel:
        store_path = path.join(args.logdir, "model")
        logging.info(f"Store pretrain model to {store_path}")

        pathlib.Path(store_path).mkdir(parents=True, exist_ok=True)
        torch.save(model_pl.pretrain_model,
            path.join(store_path, "pretrain_model.pth"))

    # ========================
    # Validation
    # ========================
    # compute final scores
    text_targets = args.text_target_loss > 0.0

    t2n = lambda x: x.detach().cpu().numpy()
    pred_dict = {"d2d": [], "p2d": [], "d2p": [], "p2p": [], "target_d": [], "target_p": []}
    model_pl.to(device)
    model_pl.eval()
    with_text = (args.notes_concepts is not None)
    logging.info("Running final validation")
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = tuple(t.to(device) for t in batch)

            if with_text:
                if text_targets:
                    tokens_d, tokens_p, tokens_text, targets_d, targets_p, _ = batch
                    logits, _ = model_pl((tokens_d, tokens_p, tokens_text))
                else:
                    tokens_d, tokens_p, tokens_text, targets_d, targets_p = batch
                    logits, _ = model_pl((tokens_d, tokens_p, tokens_text))   
            else:
                tokens_d, tokens_p, targets_d, targets_p = batch
                logits, _ = model_pl((tokens_d, tokens_p))

            preds_d2d, preds_p2d, preds_p2p, preds_d2p = tuple(torch.sigmoid(l) for l in logits)

            pred_dict["d2d"].append(t2n(preds_d2d))
            pred_dict["p2d"].append(t2n(preds_p2d))
            pred_dict["p2p"].append(t2n(preds_p2p))
            pred_dict["d2p"].append(t2n(preds_d2p))

            pred_dict["target_d"].append(t2n(targets_d))
            pred_dict["target_p"].append(t2n(targets_p))

        if args.occurrence_loss_alpha > 0.0:
            co_results = model_pl.pretrain_model.embedding.co_occurrence_loss_computation(
                split='val', compute_metrics=True, metrics_fast=False)

        if args.triplet_loss_alpha > 0.0:
            triplet_results = model_pl.pretrain_model.embedding.triplet_loss_computation(
               split='val').item()

    target_map = {"d2d": "target_d", "p2d": "target_d", "p2p": "target_p", "d2p": "target_p"}
    acc_container = {'jaccard' : 0.0, 'f1' : 0.0, 'prauc' : 0.0}

    for task, v in target_map.items():
        acc = metric_report(
            np.concatenate(pred_dict[task], axis=0), np.concatenate(pred_dict[v], axis=0),
            threshold=0.5, verbose=False)

        for k in acc_container.keys():
            acc_container[k] += acc[k]
        
        metric_string = ""
        for k in acc_container.keys():
            metric_string += f"{k}: {acc[k]:.3f} "
        logging.info(f"[{task}] {metric_string}")


    for k in acc_container.keys():
        acc_container[k] = acc_container[k] / 4.0


    logging.info(40*"=")
    logging.info(f"Finished Pretraining Training")
    logging.info(f"Trained for {model_pl.current_epoch} epochs")
    logging.info(f"Val {'Jaccard':<14}: {acc_container['jaccard']:.3f}")
    logging.info(f"Val {'F1':<14}: {acc_container['f1']:.3f}")
    logging.info(f"Val {'AuPRC':<14}: {acc_container['prauc']:.3f}")
    if args.occurrence_loss_alpha > 0.0:
        logging.info(f"Val {'CO-d AuPRC':<14}: {co_results['d_prauc']:.3f}")
        logging.info(f"Val {'CO-p AuPRC':<14}: {co_results['p_prauc']:.3f}")
    if args.triplet_loss_alpha > 0.0:
        logging.info(f"Val {'Triplet Loss':<14}: {triplet_results:.3f}")
    logging.info(40*"=")

    embedding_layer = model_pl.pretrain_model.embedding
    if args.freeze_embeddings:
        freeze_layer(embedding_layer)

    encoder_layer = model_pl.pretrain_model.encoder
    if args.freeze_encoder:
        freeze_layer(encoder_layer)

    # ========================================
    # DOWNSTREAM
    # ========================================
    downstream_gbert_acc = None
    if args.downstream_gbert:

        # clear GPU cache
        torch.cuda.empty_cache()

        if args.gbert_data is None: args.gbert_data = path.join(args.data_dir, "gbert")

        batch_size = min(int(args.batch_size // 2), 32)
        if args.down_batch_size is not None:
            batch_size = args.down_batch_size
            logging.info(f"Set downstream batchsize to {batch_size}")

        latent_space_projector = None
        if args.co_latent_space:
            latent_space_projector = model_pl.pretrain_model.latent_space_projector

        downstream_gbert_acc_val, downstream_gbert_acc = train_gbert_downstream(
                embedding_layer,
                encoder_layer,
                train_dataset.tokenizer,
                args.gbert_data,
                logger,
                max_sequence_length=MAX_SEQUENCE_LENGTH, # codes per visit
                max_visit_length=32, # visits per patient
                batch_size=batch_size,
                num_workers=args.num_workers,
                embedding_dim=args.embedding_dim,
                epochs=args.epochs,
                validation_interval=VALIDATION_INTERVAL,
                admissions_file_path=args.admission_csv,
                pretrained=args.pretrain,
                latent_space_projector=latent_space_projector,
                gated_graph_lookup=args.gated_lookup,
                attention_heads=args.attention_heads,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length,
                gradient_accumulation=args.down_grad_acc,
                es_patience=args.down_es,
                visit_model=args.gbert_visit_model,
                learning_rate=args.down_learning_rate,
                mlp_hidden_dim=(args.down_mlp_dim
                    if args.down_mlp_dim is not None else args.mlp_dim),
                mlp_num_hidden_layers=(args.down_mlp_num_layers
                    if args.down_mlp_num_layers is not None else args.mlp_num_layers),
            )

    
    downstream_cgl_acc = None
    if args.downstream_cgl:

        # clear GPU cache
        torch.cuda.empty_cache()

        batch_size = args.batch_size
        if args.down_batch_size is not None:
            batch_size = args.down_batch_size
            logging.info(f"Set downstream batchsize to {batch_size}")

        store_model_path = None
        if args.store_downmodel:
            store_model_path = path.join(args.logdir, "model")

        downstream_cgl_acc_val, downstream_cgl_acc = train_cgl_downstream(
                embedding_layer,
                encoder_layer,
                train_dataset.tokenizer,
                args.diagnosis_csv,
                args.prescription_csv,
                args.admission_csv,
                args.code_mappings,
                (train_ids, val_ids, test_ids),
                disease_codes,
                prescription_codes,
                logger,
                max_sequence_length=MAX_SEQUENCE_LENGTH, # codes per visit
                max_visit_length=42, # visits per patient
                batch_size=batch_size,
                num_workers=args.num_workers,
                embedding_dim=args.embedding_dim,
                temporal_attention_dim=args.embedding_dim,
                mlp_hidden_dim=(args.down_mlp_dim
                    if args.down_mlp_dim is not None else args.mlp_dim),
                mlp_num_hidden_layers=(args.down_mlp_num_layers
                    if args.down_mlp_num_layers is not None else args.mlp_num_layers),
                num_temporal_contexts=args.cgl_num_contexts,
                epochs=args.epochs,
                validation_interval=VALIDATION_INTERVAL,
                pretrained=args.pretrain,
                task=args.cgl_task,
                learning_rate=args.down_learning_rate,
                umls_graph_data=umls_graph,
                notes_concepts_path=args.notes_concepts,
                max_visit_text_length=args.max_visit_text_length,
                gradient_accumulation=args.down_grad_acc,
                es_patience=args.down_es,
                store_model_path=store_model_path,
                cache_graph_forward=(not args.no_cache_graph_forward),
                freeze_embedding_after=args.down_freeze_emb_after,
                eicu_dataset=args.eicu_dataset,
                down_icd_codes=args.down_icd_codes
            )
    


    # ========================================
    # WRAP-UP
    # ========================================

    # collect hyper parameters
    hyperparams = vars(args)
    hyperparams['max_seq_length'] = MAX_SEQUENCE_LENGTH

    # collect final model metrics
    final_metrics = dict()
    final_metrics['final_jaccard'] = acc_container['jaccard']
    final_metrics['final_f1'] = acc_container['f1']
    final_metrics['final_prauc'] = acc_container['prauc']

    if args.occurrence_loss_alpha > 0.0:
        final_metrics['co_prauc'] = (co_results['d_prauc'] + co_results['p_prauc']) / 2.0
    else:
        final_metrics['co_prauc'] = 0.0

    if args.triplet_loss_alpha > 0.0:
        final_metrics['triplet_loss'] = float(triplet_results)
    else:
        final_metrics['triplet_loss'] = 0.0

    if downstream_gbert_acc is not None:
        final_metrics['down_jaccard'] = downstream_gbert_acc['jaccard']
        final_metrics['down_f1'] = downstream_gbert_acc['f1']
        final_metrics['down_prauc'] = downstream_gbert_acc['prauc']
        final_metrics['down_thresh'] = downstream_gbert_acc['threshold']

        # Validation metrics
        final_metrics['down_val_prauc'] = downstream_gbert_acc_val['prauc']


    if downstream_cgl_acc is not None:
        final_metrics['down_f1'] = downstream_cgl_acc['f1']
        final_metrics['down_auroc'] = downstream_cgl_acc['auroc']
        final_metrics['down_thresh'] = downstream_cgl_acc['threshold']
        if args.cgl_task == 'diagnosis':
            final_metrics['down_f1_inflated'] = downstream_cgl_acc['f1_inflated']
            final_metrics['down_recall@20'] = downstream_cgl_acc['recall@20']
            final_metrics['down_recall@40'] = downstream_cgl_acc['recall@40']

        # Validation metrics
        final_metrics['down_val_f1'] = downstream_cgl_acc_val['f1']
        final_metrics['down_val_auroc'] = downstream_cgl_acc_val['auroc']
        if args.cgl_task == 'diagnosis':
            final_metrics['down_val_f1_inflated'] = downstream_cgl_acc_val['f1_inflated']
            final_metrics['down_val_recall@20'] = downstream_cgl_acc_val['recall@20']
            final_metrics['down_val_recall@40'] = downstream_cgl_acc_val['recall@40']

    
    # log hyperparameters
    logger.experiment.add_hparams(hyperparams, final_metrics, run_name=f"{run_name}_hparams")

    if args.store_graphmodel:
        store_path = path.join(args.logdir, "model")
        logging.info(f"Store graph to {store_path}")

        pathlib.Path(store_path).mkdir(parents=True, exist_ok=True)
        torch.save(model_pl.pretrain_model.embedding,
            path.join(store_path, "graph_embedding.pth"))

    # clean-up
    logger.finalize("success")
    logger.save()


# ========================
# SCRIPT ENTRY
# ========================
if __name__ == "__main__":
    
    # set logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s | %(message)s')
    coloredlogs.install(level=LOGGING_LEVEL)

    # run script
    main()