# ===============================================
#
# Torch CGL Modules
#
# ===============================================
import logging
from itertools import repeat
from typing import Any, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from kg.data.graph import Vocabulary
from kg.model.embedding_modules import VocabularyEmbedding
from kg.model.graph_modules import HeterogenousOntologyEmbedding
from kg.model.pretraining_modules import TransformerDecoder, TransformerEncoder
from kg.model.utility_modules import MLP
from kg.utils.metrics import roc_auc as sample_roc_auc
from kg.utils.metrics import top_k_prec_recall
from kg.utils.monitoring import get_gpu_memory_map
from kg.utils.tensors import padding_to_attention_mask, set_first_mask_entry
from kg.utils.training import freeze_layer


class CglTemporalAttention(nn.Module):
    """
    Implements simple attention module with
    context over a temporal input sequence
    """

    def __init__(self, embedding_dim: int, attention_dim: int, num_contexts: int):
        """
        Constructor for `CglTemporalAttention`

        Parameter
        ---------
        embedding_dim: -
        attention_dim: dim for the attention space
        num_contexts: number of context vectors
            if multiple they will be avg
        """
        super().__init__()

        self.num_contexts = num_contexts
        self.attention_dim = attention_dim
        self.attention_context = nn.Parameter(
            torch.zeros(attention_dim, self.num_contexts), requires_grad=True
        )

        self.project = nn.Linear(embedding_dim, attention_dim)

        # init params
        self.init_params()

    def init_params(self):
        """Initializes embedding parameters, TODO: check proper init"""
        nn.init.normal_(self.attention_context)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass

        Parameter
        ---------
        x: (B, S, E)
        mask: (B, S)

        Return
        ------
        (B, E_A)
        """

        projected = self.project(x)
        # (B, S, A)

        alignment = projected @ self.attention_context
        # (B, S, C)

        alignment = alignment.transpose(1, 2)
        # (B, C, S)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, self.num_contexts, -1))
            alignment[mask == 0] = -float("inf")
        probs = torch.softmax(alignment, dim=-1)
        # (B, C, S)

        # expand accordingly
        projected = projected.unsqueeze(1).expand((-1, self.num_contexts, -1, -1))
        probs = probs.unsqueeze(-1).expand((-1, -1, -1, self.attention_dim))
        # proj/probs: (B, C, S, A)

        # softmax aggregate
        output = torch.sum(projected * probs, dim=2)

        # avg. contexts
        output = torch.mean(output, dim=1)

        return output


class CglDownstreamModel(nn.Module):
    """
    Implements the predictive Cgl downstream task network
    """

    def __init__(
        self,
        graph_embedding: Union[HeterogenousOntologyEmbedding, VocabularyEmbedding, nn.Embedding],
        transformer: Union[TransformerDecoder, TransformerEncoder],
        padding_id: int,
        cls_id: int,
        num_classes: int,
        embedding_dim: int = 128,
        temporal_attention_dim: int = -1,
        mlp_hidden_dim: int = 64,
        mlp_dropout: float = 0.0,
        mlp_num_hidden_layers: int = 0,
        num_contexts: int = 1,
        with_text: bool = False,
        cache_graph_forward: bool = False,
    ):
        """
        Constructor for `CglDownstreamModel`

        Parameter
        ---------
        graph_embedding: from pretraining module
        transformer: from pretraining module
        padding_id: -
        cls_id: -
        num_classes: -
        embedding_dim: -
        temporal_attention_dim: dimension for the attention
            space over visits, if -1, then 2*embedding_dim
        mlp_*: parameters for final output classifier
        num_contexts: number of context vectors
            for the temporal attention over GRU
            encoded visits
        with_text: bool
            additional text token input
        cache_graph_forward: bool
            -
        """
        super().__init__()

        self.padding_id = padding_id
        self.cls_id = cls_id
        self.num_classes = num_classes
        self.graph_embedding = graph_embedding

        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.with_text = with_text
        self.latent_multiplier = 3 if self.with_text else 2
        logging.info(f"[NN] Training CGL with text: {self.with_text}")

        self.graph_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim + 1 if self.with_text else embedding_dim
        embedding_dim += 1 if self.with_text else 0

        assert isinstance(
            transformer, TransformerEncoder
        ), "Only `TransformerEncoder` supported currently"
        self.transformer = transformer

        # temporal sequence learning
        self.rnn = nn.GRU(
            input_size=self.latent_multiplier * embedding_dim,  # cat(disease, prescription)
            hidden_size=self.latent_multiplier * embedding_dim,  # keep same as input
            num_layers=1,  # simple one directional RNN
            bidirectional=False,
            batch_first=True,
            dropout=0,  # no dropout (for now)
        )

        # temporal attention
        if temporal_attention_dim == -1:
            self.temporal_attention_dim = self.latent_multiplier * embedding_dim
        else:
            self.temporal_attention_dim = temporal_attention_dim

        logging.info(f"[NN] CGL number of context vectors {num_contexts}")
        self.temporal_attention = CglTemporalAttention(
            embedding_dim * self.latent_multiplier,
            self.temporal_attention_dim,
            num_contexts,
        )

        # output classifier
        mlp_hidden_layers = list(repeat(mlp_hidden_dim, mlp_num_hidden_layers))
        mlp_layers = [self.temporal_attention_dim] + mlp_hidden_layers + [num_classes]
        logging.info(f"[NN] CGL MLP layers: {mlp_layers}")
        self.classifier = MLP(mlp_layers, dropout=mlp_dropout)

        # graph forward caching
        self.cache_graph_forward = cache_graph_forward
        logging.info(f"[NN] cache graph forward pass: {self.cache_graph_forward}")

    def forward(
        self,
        x_d,
        x_p,
        targets,
        validation: bool = False,
        x_text: torch.Tensor = None,
        fully_cache_embeddings: bool = False,
    ) -> Tuple:
        """
        Forward Pass

        Parameter
        ---------
        x_d: torch.Tensor
            batch of token ids for diseases
        x_p: torch.Tensor
            batch of token ids for prescriptions
        targets: torch.Tensor
            batch of target vectors
        validation: bool
            run in validation mode: use only last possible
            target, instead of all intermediate timesteps
        fully_cache_embeddings: bool
            also use cached values on first
            forward pass to embedding backbone i.e.
            when retrieving for diseases

        Return
        ------
        logits: torch.Tensor
            unnormalized logits, masked
        final_targets: torch.Tensor
            masked targets
        """

        # Shapes X: (B, V, C)

        # binary targets, adjust for compatibility with multi-class/label
        if len(targets.shape) == 2:
            targets = targets.unsqueeze(-1)

        # get essential dimensions
        batch_dim = x_d.shape[0]
        visit_length = x_d.shape[1]
        code_length = x_d.shape[2]
        if self.with_text:
            text_length = x_text.shape[3]  # type: ignore
            negation_mask = x_text[:, 1]  # type: ignore # second dim. negation mask
            x_text = x_text[:, 0]  # type: ignore # first dimension actual text tokens

        # get masks
        disease_mask_pad = x_d == self.padding_id
        prescr_mask_pad = x_p == self.padding_id
        if self.with_text:
            text_mask_pad = x_text == self.padding_id

        disease_mask_cls = x_d == self.cls_id
        prescr_mask_cls = x_p == self.cls_id

        disease_negation_full = torch.logical_not(
            torch.logical_or(disease_mask_pad, disease_mask_cls)
        )
        prescr_negation_full = torch.logical_not(torch.logical_or(prescr_mask_cls, prescr_mask_cls))

        # retrieve embeddings
        disease_emb = self.graph_embedding(
            x_d.view(batch_dim, -1), use_cache=fully_cache_embeddings
        )
        prescr_emb = self.graph_embedding(
            x_p.view(batch_dim, -1), use_cache=self.cache_graph_forward
        )
        if self.with_text:
            text_emb = self.graph_embedding(x_text.view(batch_dim, -1), use_cache=self.cache_graph_forward)  # type: ignore
        # (B, V*C, E)

        if self.with_text:

            disease_emb = disease_emb.view(
                batch_dim, visit_length, code_length, self.graph_embedding_dim
            )
            prescr_emb = prescr_emb.view(
                batch_dim, visit_length, code_length, self.graph_embedding_dim
            )

            disease_emb = torch.cat((disease_emb, disease_negation_full.unsqueeze(-1)), dim=-1)
            prescr_emb = torch.cat((prescr_emb, prescr_negation_full.unsqueeze(-1)), dim=-1)

            text_emb = text_emb.view(batch_dim, visit_length, text_length, self.graph_embedding_dim)
            text_emb = torch.cat((text_emb, negation_mask.unsqueeze(-1)), dim=-1)

        else:
            disease_emb = disease_emb.view(batch_dim, visit_length, code_length, self.embedding_dim)
            prescr_emb = prescr_emb.view(batch_dim, visit_length, code_length, self.embedding_dim)
        # (B, V, C, E)

        # Compute attention masks
        num_attention_heads = self.transformer.encoder.layers[0].self_attn.num_heads
        disease_mask_att = set_first_mask_entry(disease_mask_pad.view(-1, code_length))  # type: ignore

        prescr_mask_att = set_first_mask_entry(prescr_mask_pad.view(-1, code_length))  # type: ignore

        if self.with_text:
            text_mask_att = set_first_mask_entry(text_mask_pad.view(-1, text_length))  # type: ignore

        # pass through transformer
        # and retrieve visit embedding (CLS at 0)
        # (B*V, C, E)
        disease_emb_view = disease_emb.view(-1, code_length, self.embedding_dim)
        disease_encoded = self.transformer(disease_emb_view, mask=disease_mask_att)[:, 0]
        prescr_emb_view = prescr_emb.view(-1, code_length, self.embedding_dim)
        prescr_encoded = self.transformer(prescr_emb_view, mask=prescr_mask_att)[:, 0]
        if self.with_text:
            text_emb_view = text_emb.view(-1, text_length, self.embedding_dim)
            text_encoded = self.transformer(text_emb_view, mask=text_mask_att)[:, 0]
            # mask=text_mask_pad.view(-1, text_length))[:, 0]  # type: ignore
        # (B*V, E)

        # reshape for sequence learning
        disease_encoded = disease_encoded.view(batch_dim, visit_length, self.embedding_dim)
        prescr_encoded = prescr_encoded.view(batch_dim, visit_length, self.embedding_dim)
        if self.with_text:
            text_encoded = text_encoded.view(batch_dim, visit_length, self.embedding_dim)
        # (B, V, E)

        # encode visits (concat disease and prescr. CLS embeddings in embedding dimension)
        if self.with_text:
            visits_encoded = torch.cat((disease_encoded, prescr_encoded, text_encoded), dim=2)
        else:
            visits_encoded = torch.cat((disease_encoded, prescr_encoded), dim=2)
        # (B, V, 2*E)

        # get visit mask (visits where first tokens is CLS are valid visits, else PAD is at index 0)
        visit_mask = torch.where(x_d[:, :, 0] == self.cls_id, 1, 0)
        # mask: (B, V)

        # pass through single layer `uni-directional` (causal) RNN
        visits_rnn, _ = self.rnn(visits_encoded)  # get all hidden states
        # (B, V, 2*E)

        # get lower triangular, expand tensors
        # to learn at every step
        lower_tri = torch.tril(torch.ones(visit_length, visit_length, device=x_d.device))  # (V, V)
        visit_mask_ext = torch.einsum("bv,wv->bwv", visit_mask, lower_tri)
        visit_mask_ext = visit_mask_ext.view((batch_dim * visit_length, visit_length))
        # (B*V, V)

        visits_rnn_ext = (
            visits_rnn.unsqueeze(1)
            .expand((-1, visit_length, -1, -1))
            .reshape((-1, visit_length, self.latent_multiplier * self.embedding_dim))
        )
        # (B*V, V, 2*E)

        # attend over the time-series
        visits_attended = self.temporal_attention(visits_rnn_ext, mask=visit_mask_ext)
        visits_attended = visits_attended.view((batch_dim, visit_length, -1))
        # (B, V, E_A)

        if validation:
            idx = torch.arange(
                0, visit_mask.shape[-1], 1, device=x_d.device
            )  # get index for each position
            tmp = visit_mask * idx  # mask indeces
            indices = torch.argmax(tmp, 1, keepdim=True)  # get last unmasked index
            val_visit_mask = torch.zeros(visit_mask.shape, device=x_d.device, dtype=torch.int32)
            tmp_ones = torch.ones(visit_mask.shape, device=x_d.device, dtype=torch.int32)
            val_visit_mask.scatter_(1, indices, tmp_ones)  # mask all but last valid visit
            visit_mask = val_visit_mask

        # drop to prevent information leakage for TS
        # first timepoint is not a task, last datapoint would leak
        final_targets = targets[:, 1:, :]  # drop first label
        final_seq_enc = visits_attended[:, :-1, :]  # drop last datapoint
        final_target_mask = visit_mask[:, 1:]  # drop first mask index

        # select with mask
        final_targets = final_targets[final_target_mask == 1]
        final_seq_enc = final_seq_enc[final_target_mask == 1]

        # TODO: gated graph attention

        # classify
        logits = self.classifier(final_seq_enc)

        return logits, final_targets


class CglDownstreamModel_Lightning(pl.LightningModule):

    """
    Pytorch Lightning module for training a `CglDownstreamModel`

    Attributes
    ----------
    learning_rate: -
    cgl_model: a `CglDownstreamModel`
    """

    def __init__(
        self,
        cgl_model: CglDownstreamModel,
        learning_rate: float = 5e-4,
        task: str = "heart_failure",
        with_text: bool = False,
        freeze_embedding_after: int = -1,
    ):
        """
        Constructor for `CglDownstreamModel_Lightning`
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.cgl_model = cgl_model
        self.task = task
        self.binary_task = self.task == "heart_failure"
        self.with_text = with_text

        @torch.no_grad()
        def binary_output_transform(pred, target):
            x = torch.softmax(pred, dim=-1)
            return x[:, -1], target

        @torch.no_grad()
        def multi_label_transform(pred, target):
            return torch.sigmoid(pred), target

        if self.task == "heart_failure":
            self.loss_func: nn.modules.loss._Loss = nn.CrossEntropyLoss()
            self.output_transform = binary_output_transform
        else:
            self.loss_func = nn.BCEWithLogitsLoss()
            self.output_transform = multi_label_transform
        logging.info(
            f"[CGL] task: {self.task} (binary: {self.binary_task}), loss: {self.loss_func}"
        )

        self.freeze_embedding_after = freeze_embedding_after
        if self.freeze_embedding_after >= 0:
            logging.info(
                f"[CGL] Freeze embedding module after finishing epoch: {self.freeze_embedding_after}"
            )

        # save hyperparameters
        self.save_hyperparameters(ignore=["cgl_model"])

    @staticmethod
    def inflated_f1(y_true_hot, y_pred, average="weighted"):
        """
        Inflated F1 score based on `Sherbet` (or `Chet`, `CGL`) evaluation:
        https://github.com/LuChang-CS/sherbet/blob/d1061aca108eab8e0ccbd2202460e25261fdf1d5/metrics.py#L7
        """
        y_pred = np.argsort(-y_pred, axis=-1)
        result = np.zeros_like(y_true_hot)
        for i in range(len(result)):
            true_number = np.sum(y_true_hot[i] == 1)
            result[i][y_pred[i][:true_number]] = 1
        return f1_score(y_true=y_true_hot, y_pred=result, average=average, zero_division=0)

    def compute_metrics(self, preds, targets, t: float = 0.5):
        results = {}

        # binary metrics
        preds_t = np.where(preds > t, 1, 0)
        if self.task == "heart_failure":
            results["f1"] = f1_score(targets, preds_t, zero_division=0)
            try:
                roc_score = roc_auc_score(targets, preds)
            except ValueError as e:
                logging.warning(f"[METRIC] failed to compute AuROC: {e}")
                roc_score = 0.0
            results["auroc"] = roc_score

        else:
            results["f1"] = f1_score(targets, preds_t, average="weighted", zero_division=0)
            results["f1_inflated"] = self.inflated_f1(targets, preds, average="weighted")
            results["auroc"] = sample_roc_auc(targets, preds)

            ks = [10, 20, 30, 40]
            k_precision, k_recall = top_k_prec_recall(targets, preds, ks)
            for i, k in enumerate(ks):
                results[f"recall@{k}"] = k_recall[i]

        return results

    def forward(
        self,
        x_d: torch.Tensor,
        x_p: torch.Tensor,
        targets: torch.Tensor,
        validation: bool = False,
        x_text: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.cgl_model(x_d, x_p, targets, validation=validation, x_text=x_text)

    def training_step(self, batch, batch_idx):

        # pass through model
        if self.with_text:
            tokens_d, tokens_p, tokens_text, targets = batch
            preds, targets = self(tokens_d, tokens_p, targets, x_text=tokens_text)
        else:
            tokens_d, tokens_p, targets = batch
            preds, targets = self(tokens_d, tokens_p, targets)

        if self.binary_task:
            targets = targets.squeeze(-1).long()

        loss = self.loss_func(preds, targets)

        self.log("train/batch_loss", loss, prog_bar=True, logger=False)
        return loss

    def training_epoch_end(self, outputs) -> None:
        # log average training loss
        train_loss = np.mean([x["loss"].item() for x in outputs])
        self.logger.experiment.add_scalar(
            "train_down/loss", train_loss, global_step=self.current_epoch
        )
        self.log("train/loss", train_loss, prog_bar=True, logger=False)

        if self.freeze_embedding_after >= 0 and self.current_epoch == self.freeze_embedding_after:
            logging.info(f"[CGL] Freezing embedding module at end of epoch: {self.current_epoch}")
            freeze_layer(self.cgl_model.graph_embedding)

    def validation_step(self, batch, batch_idx):

        # pass through model
        if self.with_text:
            tokens_d, tokens_p, tokens_text, targets = batch
            preds, targets = self(tokens_d, tokens_p, targets, validation=True, x_text=tokens_text)
        else:
            tokens_d, tokens_p, targets = batch
            preds, targets = self(tokens_d, tokens_p, targets, validation=True)

        if self.binary_task:
            targets = targets.squeeze(-1).long()

        loss = self.loss_func(preds, targets)
        logits, targets = self.output_transform(preds, targets)

        t2n = lambda x: x.detach().cpu().numpy()
        return {
            "loss": loss.item(),
            "probs": t2n(logits),
            "targets": t2n(targets),
        }

    def validation_epoch_end(self, outputs) -> None:

        # log average validation loss
        val_loss = np.mean([x["loss"] for x in outputs])
        self.logger.experiment.add_scalar("val_down/loss", val_loss, global_step=self.current_epoch)
        self.log("val/loss", val_loss, prog_bar=True, logger=False)

        targets = np.concatenate([x["targets"] for x in outputs], axis=0)
        probs = np.concatenate([x["probs"] for x in outputs], axis=0)

        # compute metrics, F1 and AuROC, Top-k Recall
        tresh = 0.5 if self.binary_task else 0.1
        acc_container = self.compute_metrics(probs, targets, t=tresh)

        for k, v in acc_container.items():
            self.logger.experiment.add_scalar(f"val_down/{k}", v, global_step=self.current_epoch)
            if k == "auroc":
                self.log(f"val_down/{k}", v, prog_bar=True, logger=False)

        # log gpu memory
        self.logger.experiment.add_scalar(
            "monitor_down/gpu_memory",
            float(get_gpu_memory_map()[0]),
            global_step=self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
