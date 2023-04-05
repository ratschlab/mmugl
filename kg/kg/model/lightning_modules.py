# ===============================================
#
# Torch Lightning Wrapper modules to facilitate
# simpler training
#
# ===============================================
import logging
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import AveragePrecision, F1Score, JaccardIndex

from kg.model.pretraining_modules import JointCodePretrainingTransformer
from kg.training.loss import hungarian_ce_loss
from kg.utils.metrics import metric_report
from kg.utils.monitoring import get_gpu_memory_map


class PretrainingTransformer_Lightning(pl.LightningModule):
    """
    Pytorch Lightning module for training a `PretrainingTransformer`

    Attributes
    ----------
    pretrain_model: a `PretrainingTransformer` instance
    loss: a torch loss function
    num_classes: number of target codes
    learning_rate: -
    masked_loss_alpha: weight parameter for the agg. masked loss
    """

    def __init__(
        self,
        pretrain_model: nn.Module,
        loss: torch.nn.modules.loss._Loss,
        num_classes: int,
        learning_rate: float = 1e-3,
        masked_loss_alpha: float = 0.25,
    ):
        """
        Constructor for `PretrainingTransformer_Lightning`

        Parameters
        ----------
        pretrain_model: a `PretrainingTransformer` instance
        loss: target loss to use for training
        num_classes: size of target vocabulary
        learning_rate: -
        masked_loss_alpha: weight for the agg. masked loss
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.pretrain_model = pretrain_model
        self.loss = loss
        self.masked_loss_alpha = masked_loss_alpha

        # save hyperparameters
        self.save_hyperparameters(ignore=["pretrain_model", "loss"])

        # metrics
        self.acc_container: Dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pretrain_model(x)

    def training_step(self, batch, batch_idx):

        tokens, targets = batch
        logits, logits_masked = self(tokens)

        loss_cls = self.loss(logits, targets)
        loss_masked = self.loss(logits_masked, targets)
        loss = loss_cls + self.masked_loss_alpha * loss_masked

        self.log("train/batch_loss", loss, prog_bar=True, logger=False)

        return {
            "loss": loss,
            "loss_cls": loss_cls.item(),
            "loss_masked": loss_masked.item(),
        }

    def training_epoch_end(self, outputs) -> None:
        # log average training loss
        train_loss = np.mean([x["loss"].item() for x in outputs])
        self.logger.experiment.add_scalar("train/loss", train_loss, global_step=self.current_epoch)
        self.log("train/loss", train_loss, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar(
            "train/loss_cls",
            np.mean([x["loss_cls"] for x in outputs]),
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "train/loss_masked",
            np.mean([x["loss_masked"] for x in outputs]),
            global_step=self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        tokens, targets = batch
        logits, logits_masked = self(tokens)
        pred_target = torch.sigmoid(logits)

        loss_cls = self.loss(logits, targets)
        loss_masked = self.loss(logits_masked, targets)
        loss = loss_cls + self.masked_loss_alpha * loss_masked

        return {
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_masked": loss_masked.item(),
            "preds": pred_target.detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
        }

    def validation_epoch_end(self, outputs) -> None:

        # log average validation loss
        val_loss = np.mean([x["loss"] for x in outputs])
        self.logger.experiment.add_scalar("val/loss", val_loss, global_step=self.current_epoch)
        self.log("val/loss", val_loss, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar(
            "val/loss_cls",
            np.mean([x["loss_cls"] for x in outputs]),
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "val/loss_masked",
            np.mean([x["loss_masked"] for x in outputs]),
            global_step=self.current_epoch,
        )

        targets = np.concatenate([x["targets"] for x in outputs], axis=0)
        preds = np.concatenate([x["preds"] for x in outputs], axis=0)

        acc_container = metric_report(preds, targets, threshold=0.5, verbose=False, fast=True)

        for k, v in acc_container.items():
            self.logger.experiment.add_scalar(f"val/{k}", v, global_step=self.current_epoch)
            if k == "jaccard":
                self.log(f"val/{k}", v, prog_bar=True, logger=False)

        # log gpu memory
        self.logger.experiment.add_scalar(
            "monitor/gpu_memory",
            float(get_gpu_memory_map()[0]),
            global_step=self.current_epoch,
        )

        # update accuracy container
        self.acc_container = acc_container

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class JointCodePretrainingTransformer_Lightning(pl.LightningModule):
    """
    Pytorch Lightning module for training a `JointCodePretrainingTransformer`

    Attributes
    ----------
    pretrain_model: a `JointCodePretrainingTransformer` instance
    loss: a torch loss function
    num_classes: number of target codes
    learning_rate: -
    masked_loss_alpha: weight parameter for the agg. masked loss
    """

    def __init__(
        self,
        pretrain_model: JointCodePretrainingTransformer,
        loss: torch.nn.modules.loss._Loss,
        learning_rate: float = 5e-4,
        masked_loss_alpha: float = 0.25,
        set_loss_alpha: float = 0.0,
        occurrence_loss_alpha: float = 0.0,
        triplet_loss_alpha: float = 0.0,
        triplet_samples: int = 4,
        batch_size: int = 32,
        with_text: bool = False,
        text_target_loss: float = 0.0,
        no_2p_task_loss: bool = False,
    ):
        """
        Constructor for `PretrainingTransformer_Lightning`

        Parameters
        ----------
        pretrain_model: a `PretrainingTransformer` instance
        loss: target loss to use for training
        num_classes: size of target vocabulary
        learning_rate: -
        masked_loss_alpha: weight for the agg. masked loss
        set_loss_alpha: weight for the hungarian matching set loss
        occurrence_loss_alpha: weight for the co-occurrence node loss,
            model needs to be configured appropriately
        triplet_loss_alpha: -
        triplet_samples: number of triplets to sample per anchor id
        with_text: bool
            additional text token input
        text_target_loss: float
            if > 0.0, then we use additional text target loss
        no_2p_task_loss: bool
            mask out loss signal for {x}2p encoding tasks
            to focus on better disease encodings
        """
        super().__init__()

        self.with_text = with_text
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrain_model = pretrain_model
        self.loss = loss
        self.no_2p_task_loss = no_2p_task_loss
        if self.no_2p_task_loss:
            logging.warning(f"No 2p task loss: {self.no_2p_task_loss}")

        if masked_loss_alpha > 0.0:
            assert (masked_loss_alpha == 0.0) != (
                set_loss_alpha == 0.0
            ), "Only one additional loss supported"
        self.masked_loss_alpha = masked_loss_alpha
        self.set_loss_alpha = set_loss_alpha
        self.occurrence_loss_alpha = occurrence_loss_alpha
        self.triplet_loss_alpha = triplet_loss_alpha

        self.text_targets = text_target_loss > 0.0
        self.text_target_alpha = text_target_loss
        if self.text_targets:
            logging.info(f"[TRAINER] text target alpha: {self.text_target_alpha}")

        self.triplet_samples = triplet_samples
        logging.info(f"[TRAINER] Sampling {self.triplet_samples} triplets per anchor id")

        # save hyperparameters
        self.save_hyperparameters(ignore=["pretrain_model", "loss"])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pretrain_model(x)

    def training_step(self, batch, batch_idx):

        # pass through model
        if self.with_text:
            if self.text_targets:
                tokens_d, tokens_p, tokens_text, targets_d, targets_p, targets_text = batch
                logits, logits_additional = self((tokens_d, tokens_p, tokens_text))
                logits_masked, logits_text = logits_additional
            else:
                tokens_d, tokens_p, tokens_text, targets_d, targets_p = batch
                logits, logits_masked = self((tokens_d, tokens_p, tokens_text))
        else:
            tokens_d, tokens_p, targets_d, targets_p = batch
            logits, logits_masked = self((tokens_d, tokens_p))

        # compute cls reconstruction clf loss
        logits_d2d, logits_p2d, logits_p2p, logits_d2p = logits

        # 2p factor
        prescription_loss_multiplier = 1.0
        if self.no_2p_task_loss:
            prescription_loss_multiplier = 0.0

        loss_cls = (
            self.loss(logits_d2d, targets_d)
            + self.loss(logits_p2d, targets_d)
            + self.loss(logits_p2p, targets_p) * prescription_loss_multiplier
            + self.loss(logits_d2p, targets_p) * prescription_loss_multiplier
        )

        # token aggregation loss
        loss_additional = torch.tensor(0.0)
        if self.masked_loss_alpha > 0:
            disease_logits_masked, prescr_logits_masked = logits_masked

            loss_masked = (
                self.loss(disease_logits_masked, targets_d)
                + self.loss(prescr_logits_masked, targets_p) * prescription_loss_multiplier
            )

            loss_additional = self.masked_loss_alpha * loss_masked

        # hungarian set matching loss
        elif self.set_loss_alpha > 0:
            disease_logits, prescr_logits, disease_mask, prescr_mask = logits_masked

            disease_loss = hungarian_ce_loss(disease_logits, targets_d, x_mask=disease_mask)
            prescr_loss = hungarian_ce_loss(prescr_logits, targets_p, x_mask=prescr_mask)

            loss_additional = self.set_loss_alpha * (disease_loss + prescr_loss)

        # add additional loss on disease/prescription
        loss = loss_cls + loss_additional

        # add text loss
        loss_text_cpu = 0
        if self.text_targets:
            loss_text = self.loss(logits_text, targets_text)
            loss += self.text_target_alpha * loss_text
            loss_text_cpu = loss_text.item()

        # apply co-occurrence loss if appicable
        if self.occurrence_loss_alpha > 0.0:

            # compute loss on random batch of co-occurrence nodes
            co_results = self.pretrain_model.embedding.co_occurrence_loss_computation(
                split="train", compute_metrics=False, batch=True, return_embeddings=True
            )

            # alpha considered in graph module
            loss += co_results["loss"]

        # apply triplet loss if appicable
        if self.triplet_loss_alpha > 0.0:

            triplet_results = self.pretrain_model.embedding.triplet_loss_computation(
                split="train",
                connection_nodes="disease",
                samples=self.triplet_samples,
                graph_embeddings=co_results["graph_embeddings"]
                # NOTE: this adopts the co dropout, but saves compute
            )

            loss += self.pretrain_model.embedding.triplet_loss_alpha * triplet_results

        self.log("train/batch_loss", loss, prog_bar=True, logger=False)

        data_dict = {
            "loss": loss,
            "loss_cls": loss_cls.item(),
            "loss_additional": loss_additional.item(),
        }

        if self.text_targets:
            data_dict["loss_text"] = loss_text_cpu

        # log co loss
        if self.occurrence_loss_alpha > 0.0:
            data_dict["loss_co"] = co_results["loss"].item()

        # log triplet loss
        if self.triplet_loss_alpha > 0.0:
            data_dict["loss_triplet"] = triplet_results.item()

        return data_dict

    def training_epoch_end(self, outputs) -> None:
        # log average training loss
        train_loss = np.mean([x["loss"].item() for x in outputs])
        self.logger.experiment.add_scalar("train/loss", train_loss, global_step=self.current_epoch)
        self.log("train/loss", train_loss, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar(
            "train/loss_cls",
            np.mean([x["loss_cls"] for x in outputs]),
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "train/loss_additional",
            np.mean([x["loss_additional"] for x in outputs]),
            global_step=self.current_epoch,
        )

        # log text loss
        if self.text_targets:
            self.logger.experiment.add_scalar(
                "train/loss_text",
                np.mean([x["loss_text"] for x in outputs]),
                global_step=self.current_epoch,
            )

        # log co loss
        if self.occurrence_loss_alpha > 0.0:
            self.logger.experiment.add_scalar(
                "train/loss_co",
                np.mean([x["loss_co"] for x in outputs]),
                global_step=self.current_epoch,
            )

        # log triplet loss
        if self.triplet_loss_alpha > 0.0:
            self.logger.experiment.add_scalar(
                "train/loss_triplet",
                np.mean([x["loss_triplet"] for x in outputs]),
                global_step=self.current_epoch,
            )

    def validation_step(self, batch, batch_idx):
        # pass through model
        if self.with_text:
            if self.text_targets:
                tokens_d, tokens_p, tokens_text, targets_d, targets_p, targets_text = batch
                logits, logits_additional = self((tokens_d, tokens_p, tokens_text))
                logits_masked, logits_text = logits_additional
            else:
                tokens_d, tokens_p, tokens_text, targets_d, targets_p = batch
                logits, logits_masked = self((tokens_d, tokens_p, tokens_text))
        else:
            tokens_d, tokens_p, targets_d, targets_p = batch
            logits, logits_masked = self((tokens_d, tokens_p))

        # compute loss and probabilites
        logits_d2d, logits_p2d, logits_p2p, logits_d2p = logits
        preds_d2d, preds_p2d, preds_p2p, preds_d2p = tuple(torch.sigmoid(l) for l in logits)

        # 2p factor
        prescription_loss_multiplier = 1.0
        if self.no_2p_task_loss:
            prescription_loss_multiplier = 0.0

        loss_cls = (
            self.loss(logits_d2d, targets_d)
            + self.loss(logits_p2d, targets_d)
            + self.loss(logits_p2p, targets_p) * prescription_loss_multiplier
            + self.loss(logits_d2p, targets_p) * prescription_loss_multiplier
        )

        # token aggregation loss
        loss_additional = torch.tensor(0.0)
        if self.masked_loss_alpha > 0:
            disease_logits_masked, prescr_logits_masked = logits_masked

            loss_masked = (
                self.loss(disease_logits_masked, targets_d)
                + self.loss(prescr_logits_masked, targets_p) * prescription_loss_multiplier
            )

            loss_additional = self.masked_loss_alpha * loss_masked

        # hungarian set matching loss
        elif self.set_loss_alpha > 0:
            disease_logits, prescr_logits, disease_mask, prescr_mask = logits_masked

            disease_loss = hungarian_ce_loss(disease_logits, targets_d, x_mask=disease_mask)
            prescr_loss = hungarian_ce_loss(prescr_logits, targets_p, x_mask=prescr_mask)

            loss_additional = self.set_loss_alpha * (disease_loss + prescr_loss)

        # add additional loss on disease/prescription
        loss = loss_cls + loss_additional

        # add text loss
        loss_text_cpu = 0
        if self.text_targets:
            loss_text = self.loss(logits_text, targets_text)
            loss += self.text_target_alpha * loss_text
            loss_text_cpu = loss_text.item()

        t2n = lambda x: x.detach().cpu().numpy()
        data_dict = {
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_additional": loss_additional.item(),
            "targets_d": t2n(targets_d),
            "targets_p": t2n(targets_p),
            "preds_d2d": t2n(preds_d2d),
            "preds_p2d": t2n(preds_p2d),
            "preds_p2p": t2n(preds_p2p),
            "preds_d2p": t2n(preds_d2p),
        }

        if self.text_targets:
            data_dict["loss_text"] = loss_text_cpu

        return data_dict

    def validation_epoch_end(self, outputs) -> None:

        # log average validation loss
        val_loss = np.mean([x["loss"] for x in outputs])
        self.logger.experiment.add_scalar("val/loss", val_loss, global_step=self.current_epoch)
        self.log("val/loss", val_loss, prog_bar=True, logger=False)

        self.logger.experiment.add_scalar(
            "val/loss_cls",
            np.mean([x["loss_cls"] for x in outputs]),
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_scalar(
            "val/loss_additional",
            np.mean([x["loss_additional"] for x in outputs]),
            global_step=self.current_epoch,
        )

        if self.text_targets:
            self.logger.experiment.add_scalar(
                "val/loss_text",
                np.mean([x["loss_text"] for x in outputs]),
                global_step=self.current_epoch,
            )

        targets_d = np.concatenate([x["targets_d"] for x in outputs], axis=0)
        targets_p = np.concatenate([x["targets_p"] for x in outputs], axis=0)

        pred_dict = {
            "d2d": ["preds_d2d", targets_d],
            "p2d": ["preds_p2d", targets_d],
            "d2p": ["preds_d2p", targets_p],
            "p2p": ["preds_p2p", targets_p],
        }

        acc_container = {}
        for task, v in pred_dict.items():
            preds = np.concatenate([x[v[0]] for x in outputs], axis=0)
            acc_container[task] = metric_report(
                preds, v[1], threshold=0.5, verbose=False, fast=True
            )

        for task, acc in acc_container.items():
            for k, value in acc.items():
                self.logger.experiment.add_scalar(
                    f"val/{task}/{k}", value, global_step=self.current_epoch
                )

        # log gpu memory
        self.logger.experiment.add_scalar(
            "monitor/gpu_memory",
            float(get_gpu_memory_map()[0]),
            global_step=self.current_epoch,
        )

        # total loss
        total_loss = val_loss

        # co occurence performance
        if self.occurrence_loss_alpha > 0.0:

            # compute loss on co-occurrence nodes
            results = self.pretrain_model.embedding.co_occurrence_loss_computation(  # type: ignore
                split="val", compute_metrics=True, batch=False
            )
            total_loss += results["loss"]

            # log
            for k, value in results.items():
                self.logger.experiment.add_scalar(
                    f"co_val/{k}", value, global_step=self.current_epoch
                )

        # triplet performance
        if self.triplet_loss_alpha > 0.0:
            results = self.pretrain_model.embedding.triplet_loss_computation(split="val")  # type: ignore
            self.logger.experiment.add_scalar(
                f"val/loss_triplet", results.item(), global_step=self.current_epoch
            )
            total_loss += results.item()

        # total loss
        self.logger.experiment.add_scalar(
            "val/loss_total", total_loss, global_step=self.current_epoch
        )
        self.log("val/loss_total", total_loss, prog_bar=True, logger=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
