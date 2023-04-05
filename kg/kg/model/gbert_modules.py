# ===============================================
#
# Torch Gbert Modules
#
# ===============================================
import logging
from itertools import repeat
from typing import Any, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from kg.data.graph import Vocabulary
from kg.model.embedding_modules import VocabularyEmbedding
from kg.model.graph_modules import HeterogenousOntologyEmbedding
from kg.model.pretraining_modules import (
    TransformerDecoder,
    TransformerEncoder,
    VisitLatentSpaceProjector,
)
from kg.model.utility_modules import MLP, GatedGraphLookup
from kg.utils.metrics import metric_report
from kg.utils.monitoring import get_gpu_memory_map
from kg.utils.tensors import padding_to_attention_mask, set_first_mask_entry


class GbertVisitModel(nn.Module):
    """
    Implements a sequence learning model
    over a series of visits and returns
    a representation for each step
    """

    def __init__(
        self,
        embedding_dim: int,
    ):

        super().__init__()
        self.embedding_dim = embedding_dim


class GbertVisitAverage(GbertVisitModel):
    """
    Computes for each time-step an average over
    the past
    """

    def __init__(self, embedding_dim: int):

        super().__init__(embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, V, E)

        # average aggregate visits
        visit_length = x.shape[1]
        lower_tri = torch.tril(torch.ones(visit_length, visit_length, device=x.device))  # (V, V)
        div = (
            torch.sum(lower_tri, dim=-1).unsqueeze(-1).expand(visit_length, self.embedding_dim)
        )  # (V, E)

        x_agg = torch.div(torch.matmul(lower_tri, x), div)
        return x_agg


class GbertVisitRNN(GbertVisitModel):
    """
    GRU over the past
    """

    def __init__(self, embedding_dim: int, latent_dim: int = -1, dropout: float = 0):

        super().__init__(embedding_dim=embedding_dim)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=(embedding_dim if latent_dim == -1 else latent_dim),
            num_layers=1,
            batch_first=True,  # (B, V, E)
            dropout=dropout,
            bidirectional=False,  # causal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, V, E)

        x_agg, _ = self.gru(x)
        # (B, V, E)

        return x_agg


class GbertDownstreamModel(nn.Module):
    """
    Implements the predictive Gbert downstream task network
    """

    def __init__(
        self,
        graph_embedding: Union[HeterogenousOntologyEmbedding, VocabularyEmbedding, nn.Embedding],
        transformer: Union[TransformerDecoder, TransformerEncoder],
        padding_id: int,
        cls_id: int,
        num_classes: int,
        embedding_dim: int = 128,
        mlp_hidden_dim: int = 64,
        mlp_dropout: float = 0.0,
        mlp_num_hidden_layers: int = 2,
        latent_space_projector: VisitLatentSpaceProjector = None,
        gated_graph_lookup: List[str] = None,
        attention_heads: int = 1,
        with_text: bool = False,
        visit_model: str = "avg",
    ):
        super().__init__()

        self.padding_id = padding_id
        self.cls_id = cls_id
        self.num_classes = num_classes
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.with_text = with_text

        self.graph_embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim + 1 if self.with_text else embedding_dim
        embedding_dim += 1 if self.with_text else 0

        self.graph_embedding = graph_embedding

        assert isinstance(
            transformer, TransformerEncoder
        ), "Only `TransformerEncoder` supported currently"
        self.transformer = transformer

        # Visit encoding model
        self.visit_model_dim = 3 * embedding_dim if self.with_text else 2 * embedding_dim
        self.visit_model: GbertVisitModel
        if visit_model == "avg":
            logging.info(f"[GBERT] visit model: past average")
            self.visit_model = GbertVisitAverage(self.visit_model_dim)
        elif visit_model == "rnn":
            logging.info(f"[GBERT] visit model: RNN")
            self.visit_model = GbertVisitRNN(
                embedding_dim=self.visit_model_dim, latent_dim=embedding_dim
            )
            self.visit_model_dim = embedding_dim
        elif visit_model == "rnn-wide":
            logging.info(f"[GBERT] visit model: RNN wide")
            self.visit_model = GbertVisitRNN(
                embedding_dim=self.visit_model_dim, latent_dim=self.visit_model_dim
            )
        else:
            raise ValueError(f"[GBERT] unkown visit model config: {visit_model}")

        self.classifier_input_dim = self.visit_model_dim + embedding_dim
        classifier_dims = (
            [self.classifier_input_dim]
            + list(repeat(mlp_hidden_dim, mlp_num_hidden_layers))
            + [num_classes]
        )
        logging.info(f"[GBERT] classifier dims: {classifier_dims}")
        self.classifier = MLP(
            classifier_dims,
            dropout=mlp_dropout,
        )

    def forward(self, x_d, x_p, targets_p, x_text: torch.Tensor = None):

        # Shapes X: (B, V, C)

        # get essential dimensions
        batch_dim = x_d.shape[0]
        visit_length = x_d.shape[1]
        code_length = x_d.shape[2]
        if self.with_text:
            text_length = x_text.shape[3]  # type: ignore
            negation_mask = x_text[:, 1]  # type: ignore # second dim. negation mask
            x_text = x_text[:, 0]  # type: ignore # first dimension actual text tokens

        # retrieve embeddings
        disease_emb = self.graph_embedding(x_d.view(batch_dim, -1), use_cache=False)
        prescr_emb = self.graph_embedding(x_p.view(batch_dim, -1), use_cache=True)
        if self.with_text:
            text_emb = self.graph_embedding(x_text.view(batch_dim, -1), use_cache=True)  # type: ignore
        # (B, V*C, E)

        disease_emb = disease_emb.view(
            batch_dim, visit_length, code_length, self.graph_embedding_dim
        )
        prescr_emb = prescr_emb.view(batch_dim, visit_length, code_length, self.graph_embedding_dim)
        # (B, V, C, E)

        # get masks
        disease_mask_pad = x_d == self.padding_id
        prescr_mask_pad = x_p == self.padding_id
        if self.with_text:
            text_mask_pad = x_text == self.padding_id

        disease_mask_cls = x_d == self.cls_id
        prescr_mask_cls = x_p == self.cls_id

        # compute negation features for codes (positive for each code)
        disease_negation_full = torch.logical_not(
            torch.logical_or(disease_mask_pad, disease_mask_cls)
        )
        prescr_negation_full = torch.logical_not(torch.logical_or(prescr_mask_cls, prescr_mask_cls))

        # reshape and add negation features
        if self.with_text:

            # cat code negation features
            disease_emb = torch.cat((disease_emb, disease_negation_full.unsqueeze(-1)), dim=-1)
            prescr_emb = torch.cat((prescr_emb, prescr_negation_full.unsqueeze(-1)), dim=-1)

            # cat text negation features
            text_emb = text_emb.view(batch_dim, visit_length, text_length, self.graph_embedding_dim)
            text_emb = torch.cat((text_emb, negation_mask.unsqueeze(-1)), dim=-1)

            # reshape for encoding
            disease_emb = disease_emb.view(-1, code_length, self.embedding_dim)
            prescr_emb = prescr_emb.view(-1, code_length, self.embedding_dim)
            text_emb = text_emb.view(-1, text_length, self.embedding_dim)
            # (B*V, C, E)

        else:
            disease_emb = disease_emb.view(-1, code_length, self.embedding_dim)
            prescr_emb = prescr_emb.view(-1, code_length, self.embedding_dim)
            # (B*V, C, E)

        # Compute attention masks
        num_attention_heads = self.transformer.encoder.layers[0].self_attn.num_heads
        disease_mask_att = set_first_mask_entry(disease_mask_pad.view(-1, code_length))  # type: ignore
        # disease_mask_att = padding_to_attention_mask(
        #     disease_mask_att, num_heads=num_attention_heads)

        prescr_mask_att = set_first_mask_entry(prescr_mask_pad.view(-1, code_length))  # type: ignore
        # prescr_mask_att = padding_to_attention_mask(
        #     prescr_mask_att, num_heads=num_attention_heads)

        if self.with_text:
            text_mask_att = set_first_mask_entry(text_mask_pad.view(-1, text_length))  # type: ignore
            # text_mask_att = padding_to_attention_mask(
            #     text_mask_att, num_heads=num_attention_heads)

        # pass through transformer
        # (B*V, C, E)
        disease_encoded = self.transformer(disease_emb, mask=disease_mask_att)[:, 0]
        prescr_encoded = self.transformer(prescr_emb, mask=prescr_mask_att)[:, 0]
        if self.with_text:
            text_encoded = self.transformer(text_emb, mask=text_mask_att)[:, 0]
        # (B*V, E)

        # reshape for sequence learning
        disease_encoded = disease_encoded.view(batch_dim, visit_length, self.embedding_dim)
        prescr_encoded = prescr_encoded.view(batch_dim, visit_length, self.embedding_dim)
        if self.with_text:
            text_encoded = text_encoded.view(batch_dim, visit_length, self.embedding_dim)
        # (B, V, E)


        # (B, V, E)
        all_encodings: Union[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ]
        if self.with_text:
            all_encodings = (disease_encoded, prescr_encoded, text_encoded)
        else:
            all_encodings = (disease_encoded, prescr_encoded)
        all_encoded = torch.cat(all_encodings, dim=-1)

        # visit modeling
        all_agg = self.visit_model(all_encoded)

        # mask visits without targets
        target_sum = torch.sum(targets_p, dim=-1)
        target_mask = torch.where(target_sum[:, 1:] == 0, 0, 1).view(-1)

        # reshape and drop unneeded
        disease_encoded_cut = torch.reshape(disease_encoded[:, 1:], (-1, self.embedding_dim))
        all_agg_cut = torch.reshape(all_agg[:, :-1], (-1, self.visit_model_dim))

        # concatenate features
        features_concat = torch.cat((disease_encoded_cut, all_agg_cut), dim=1)

        # compute logits
        logits = self.classifier(features_concat)
        targets = torch.reshape(targets_p[:, 1:], (-1, self.num_classes))

        # compute individual losses (aggregate across individual classes)
        per_sample_loss = torch.mean(self.loss(logits, targets), dim=1)

        # mask loss
        masked_loss = per_sample_loss * target_mask

        # aggregate loss
        agg_loss = torch.sum(masked_loss) / torch.sum(target_mask)

        return agg_loss, logits, targets, target_mask


class GbertDownstreamModel_Lightning(pl.LightningModule):
    """
    Pytorch Lightning module for training a `GbertDownstreamModel`

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
        gbert_model: nn.Module,
        learning_rate: float = 5e-4,
        with_text: bool = False,
    ):
        """
        Constructor for `GbertDownstreamModel_Lightning`
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.gbert_model = gbert_model
        self.with_text = with_text

        # save hyperparameters
        self.save_hyperparameters(ignore=["gbert_model"])

    def forward(
        self,
        x_d: torch.Tensor,
        x_p: torch.Tensor,
        y_p: torch.Tensor,
        x_text: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.gbert_model(x_d, x_p, y_p, x_text=x_text)

    def training_step(self, batch, batch_idx):

        # pass through model
        if self.with_text:
            tokens_d, tokens_p, tokens_text, targets_p = batch
            loss, _, _, _ = self(tokens_d, tokens_p, targets_p, x_text=tokens_text)
        else:
            tokens_d, tokens_p, targets_p = batch
            loss, _, _, _ = self(tokens_d, tokens_p, targets_p)

        self.log("train/batch_loss", loss, prog_bar=True, logger=False)
        return loss

    def training_epoch_end(self, outputs) -> None:
        # log average training loss
        train_loss = np.mean([x["loss"].item() for x in outputs])
        self.logger.experiment.add_scalar(
            "train_down/loss", train_loss, global_step=self.current_epoch
        )
        self.log("train/loss", train_loss, prog_bar=True, logger=False)

    def validation_step(self, batch, batch_idx):

        # pass through model
        if self.with_text:
            tokens_d, tokens_p, tokens_text, targets_p = batch
            loss, logits, targets, target_mask = self(
                tokens_d, tokens_p, targets_p, x_text=tokens_text
            )
        else:
            tokens_d, tokens_p, targets_p = batch
            loss, logits, targets, target_mask = self(tokens_d, tokens_p, targets_p)
        logits = torch.sigmoid(logits)

        # collect unmasked logits, targets
        collect_logits = []
        collect_targets = []
        for logit, target, mask in zip(logits, targets, target_mask):
            if mask == 1:
                collect_logits.append(logit)
                collect_targets.append(target)

        collect_logits = torch.stack(collect_logits)
        collect_targets = torch.stack(collect_targets)

        t2n = lambda x: x.detach().cpu().numpy()
        return {
            "loss": loss.item(),
            "logits": t2n(collect_logits),
            "targets": t2n(collect_targets),
        }

    def validation_epoch_end(self, outputs) -> None:

        # log average validation loss
        val_loss = np.mean([x["loss"] for x in outputs])
        self.logger.experiment.add_scalar("val_down/loss", val_loss, global_step=self.current_epoch)
        self.log("val/loss", val_loss, prog_bar=True, logger=False)

        targets = np.concatenate([x["targets"] for x in outputs], axis=0)
        logits = np.concatenate([x["logits"] for x in outputs], axis=0)

        acc_container = metric_report(logits, targets, threshold=0.5, verbose=False, fast=True)

        for k, v in acc_container.items():
            self.logger.experiment.add_scalar(f"val_down/{k}", v, global_step=self.current_epoch)
            if k == "jaccard":
                self.log(f"val/{k}", v, prog_bar=True, logger=False)

        # log gpu memory
        self.logger.experiment.add_scalar(
            "monitor_down/gpu_memory",
            float(get_gpu_memory_map()[0]),
            global_step=self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
