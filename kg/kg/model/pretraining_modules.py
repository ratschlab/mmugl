# ===============================================
#
# Torch Pretraining Modules
#
# ===============================================
import logging
from itertools import repeat
from turtle import forward
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn

from kg.data.datasets import CodeTokenizer
from kg.data.graph import CoLinkConfig, Vocabulary
from kg.model.embedding_modules import FlatEmbedding, VocabularyEmbedding
from kg.model.graph_modules import HeterogenousOntologyEmbedding
from kg.model.umls_modules import UMLSGraphEmbedding
from kg.model.utility_modules import MLP, AsymmetricMemoryAttention
from kg.utils.tensors import padding_to_attention_mask, set_first_mask_entry


class TransformerEncoder(nn.Module):
    """
    Transformer Module for sequence encoding

    Attributes
    ----------
    encoder: transformer encoder of `num_blocks` blocks
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        attention_heads: int = 1,
        feedforward_dim: int = 512,
        activation: str = "gelu",
        num_blocks: int = 1,
        layer_norm_eps: float = 1e-08,
    ):
        """
        Constructor for `TransformerEncoder`

        Parameters
        ----------
        hidden_dim: hidden dimension of the transformer
        attention_heads: number of attention heads
        feedforward_dim: mlp dim of the transformer; usually 4*hidden_dim
        activation: good options: {"relu", "gelu"}
        num_blocks: number of stacked transformer blocks
        layer_norm_eps: layer_norm_eps of the Transformer module
        """
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            activation=activation,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
            dropout=0.1,
        )
        encoder_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_blocks, encoder_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_encoded = self.encoder(
            x, src_key_padding_mask=mask
        )  # x_encoded = self.encoder(x, mask=mask)
        x_encoded = torch.nan_to_num(x_encoded, nan=0.0)
        return x_encoded


class TransformerDecoder(nn.Module):
    """
    Transformer Module for sequence encoding
    in decoding style (attend to graph embeddings)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        attention_heads: int = 1,
        feedforward_dim: int = 512,
        activation: str = "gelu",
        num_blocks: int = 1,
        layer_norm_eps: float = 1e-08,
    ):
        """
        Constructor for `TransformerDecoder`

        Parameters
        ----------
        hidden_dim: hidden dimension of the transformer
        attention_heads: number of attention heads
        feedforward_dim: mlp dim of the transformer; usually 4*hidden_dim
        activation: good options: {"relu", "gelu"}
        num_blocks: number of stacked transformer blocks
        layer_norm_eps: layer_norm_eps of the Transformer module
        """
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            activation=activation,
            batch_first=True,
            layer_norm_eps=layer_norm_eps,
            dropout=0.1,
        )
        decoder_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_blocks, decoder_norm)

    def forward(
        self, x: torch.Tensor, graph_embeddings: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:

        return self.decoder(x, graph_embeddings, tgt_key_padding_mask=mask)


class VisitPredictionHead(nn.Module):
    """
    Prediction head for code predictions
    based on the CLS token

    Attributes
    ----------
    classifier: mlp computes vocabulary logits
    """

    def __init__(
        self,
        vocabulary_size: int,
        input_dim: int = 128,
        mlp_hidden_dims: List[int] = [128],
    ):
        """
        Constructor for `VisitPredictionHead`

        Parameters
        ----------
        vocabulary_size: size of the output vocabulary
        input_dim: input dimension
        mlp_hidden_dim: list of hidden dimension of the MLP
        """
        super(VisitPredictionHead, self).__init__()

        dims = [input_dim] + mlp_hidden_dims + [vocabulary_size]
        self.classifier = MLP(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MaskedAggregatorPredictionHead(nn.Module):
    """
    Prediction head for code predictions by
    predicting logits for each masked token
    and then aggregating the logits for all
    non-masked samples
    """

    def __init__(
        self,
        vocabulary_size: int,
        input_dim: int = 128,
        mlp_hidden_dims: List[int] = [128],
    ):
        """
        Constructor for `MaskedAggregatorPredictionHead`

        Parameters
        ----------
        vocabulary_size: size of the output vocabulary
        input_dim: input dimension
        mlp_hidden_dim: list of hidden dimension of the MLP
        """
        super(MaskedAggregatorPredictionHead, self).__init__()

        self.vocabulary_size = vocabulary_size

        dims = [input_dim] + mlp_hidden_dims + [vocabulary_size]
        self.classifier = MLP(dims)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Project to vocabulary size (N, S, E) -> (N, S, V)
        out = self.classifier(x)

        # mask logits where True (mask padding and cls)
        if mask is not None:
            mask = torch.unsqueeze(mask, -1).repeat(1, 1, self.vocabulary_size)
            out = torch.where(mask, torch.tensor(0, dtype=torch.float, device=x.device), out)

        # reduce in the sequence dimension (N, S, V) -> (N, V)
        out = torch.sum(out, dim=1, keepdim=False)

        return out


class VisitLatentSpaceProjector(nn.Module):
    """
    Projects a CLS encoded visit (disease, prescr) onto
    the latent space of a set of embeddings (e.g. graph nodes)
    """

    def __init__(self, embedding_dim: int, key_dim: int, value_dim: int, num_heads: int = 1):
        super(VisitLatentSpaceProjector, self).__init__()

        self.co_attention_d = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            kdim=key_dim,
            vdim=value_dim,
        )

        self.co_attention_p = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            kdim=key_dim,
            vdim=value_dim,
        )

    def forward(self, x_d, x_p, embeddings):
        """
        Forward method, batch is not first!

        Parameter
        ---------
        x_d: (S, B, E)
        x_p: (S, B, E)
        embeddings: (N, B, N_E `key_dim`)
        """

        visit_encoding_d, _ = self.co_attention_d(
            query=x_d, key=embeddings, value=embeddings, need_weights=False
        )

        visit_encoding_p, _ = self.co_attention_p(
            query=x_p, key=embeddings, value=embeddings, need_weights=False
        )
        # (S, B, EMBEDDING)

        return visit_encoding_d, visit_encoding_p


class PretrainingTransformer(nn.Module):
    """
    Transformer based model for pretraining
    using the CLS visit representation learning task
    """

    def __init__(
        self,
        disease_vocabulary: Vocabulary,
        padding_id: int,
        cls_id: int,
        num_special_tokens: int = 3,
        embedding_dim: int = 128,
        graph_num_layers: int = 1,
        graph_staged: bool = False,
        attention_heads: int = 1,
        feedforward_dim: int = 512,
        num_blocks: int = 1,
        activation: str = "gelu",
        mlp_dim: int = 128,
        mlp_num_layers: int = 1,
        agg_mask_output: bool = False,
        decoder_network: bool = False,
    ):
        """
        Constructor for `PretrainingTransformer`

        Parameters
        ----------
        disease_vocabulary: target vocabulary
        padding_id: id of padding [PAD] token for mask generation
        num_special_tokens: -
        embedding_dim: dimension of the embeddings, the graph layers,
            and the transformer
        graph_num_layers: number of graph layers
        graph_staged: use staged or undirected GNN
        attention_heads: -
        feedforward_dim: dim for the transformer MLP
        num_blocks: #transformer blocks
        activation: transformer activation {"gelu", "relu"}
        mlp_dim: classifier hiddden dim
        mlp_num_layers
        agg_mask_output: whether to project each sequence output to
            the vocabulary and output the aggregated logits additionally
        decoder_network: use a decoder network, which does attend to
            all the graph embeddings
        """
        super(PretrainingTransformer, self).__init__()

        self.padding_id = padding_id
        self.cls_id = cls_id
        self.embedding = VocabularyEmbedding(
            disease_vocabulary=disease_vocabulary,
            num_special_tokens=num_special_tokens,
            hidden_dims=list(repeat(embedding_dim, graph_num_layers)),
            disease_hidden_dim=embedding_dim,
            graph_staged=graph_staged,
        )

        self.decoder_network = decoder_network
        if decoder_network:
            self.decoder = TransformerDecoder(
                hidden_dim=embedding_dim,
                attention_heads=attention_heads,
                feedforward_dim=feedforward_dim,
                activation=activation,
                num_blocks=num_blocks,
            )
        else:
            self.encoder = TransformerEncoder(
                hidden_dim=embedding_dim,
                attention_heads=attention_heads,
                feedforward_dim=feedforward_dim,
                activation=activation,
                num_blocks=num_blocks,
            )

        self.classifier = VisitPredictionHead(
            len(disease_vocabulary.word2idx),
            input_dim=embedding_dim,
            mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
        )

        self.agg_mask_output = agg_mask_output
        if agg_mask_output:
            self.mask_classifier = MaskedAggregatorPredictionHead(
                len(disease_vocabulary.word2idx),
                input_dim=embedding_dim,
                mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # retrieve graph embeddings
        emb = self.embedding(x)

        # transformer interaction modeling
        mask_pad = x == self.padding_id
        if self.decoder_network:
            graph_embeddings = self.embedding.disease_embedding.get_all_graph_embeddings()
            graph_embeddings = torch.unsqueeze(graph_embeddings, 0).expand(x.shape[0], -1, -1)
            encoded = self.decoder(emb, graph_embeddings, mask=mask_pad)
        else:
            encoded = self.encoder(emb, mask=mask_pad)

        # classify [CLS]
        logits = self.classifier(encoded[:, 0])

        # classify and aggregate masked tokens
        if self.agg_mask_output:
            mask_cls = x == self.cls_id
            mask_full = torch.logical_or(mask_pad, mask_cls)
            logits_masked = self.mask_classifier(encoded, mask=mask_full)
            return logits, logits_masked

        return logits


class JointCodePretrainingTransformer(nn.Module):
    """
    Transformer based model for pretraining
    using the CLS visit representation learning task
    for ICD and ATC codes combined
    """

    def __init__(
        self,
        tokenizer: CodeTokenizer,
        padding_id: int,
        cls_id: int,
        num_special_tokens: int = 3,
        embedding_dim: int = 128,
        graph_num_layers: int = 1,
        graph_num_filters: int = 1,
        graph_staged: bool = False,
        attention_heads: int = 1,
        feedforward_dim: int = 512,
        num_blocks: int = 1,
        activation: str = "gelu",
        mlp_dim: int = 128,
        mlp_num_layers: int = 1,
        agg_mask_output: bool = False,
        token_logit_output: bool = False,
        decoder_network: bool = False,
        convolution_operator: str = "GCNConv",
        data_pd: Optional[pd.DataFrame] = None,
        co_occurrence: Optional[str] = None,
        co_occurrence_subsample: float = 0.3,
        co_occurrence_loss: float = 0.0,
        co_occurrence_dropout: float = 0.0,
        co_occurrence_divisor: int = 4,
        co_latent_space: bool = False,
        co_occurrence_cluster: int = 0,
        co_occurrence_features: Optional[str] = None,
        co_occurrence_batch_size: int = 16,
        triplet_loss: float = 0.0,
        triplet_batch_size: int = 16,
        triplet_margin: float = 0.1,
        graph_memory_size: int = 0,
        umls_graph: Dict = None,
        co_link_config: CoLinkConfig = None,
        with_text: bool = False,
        contractions_type: str = None,
        trainable_edge_weights: bool = False,
        cache_graph_forward: bool = False,
        text_targets: bool = False,
    ):
        """
        Constructor for `PretrainingTransformer`

        Parameters
        ----------
        tokenizer: `CodeTokenizer` associated with this network
        padding_id: id of padding [PAD] token for mask generation
        num_special_tokens: -
        embedding_dim: dimension of the embeddings, the graph layers,
            and the transformer
        graph_num_layers: number of graph layers
        graph_num_filters: int
            number of distinct stacks of graph layers
        graph_staged: use staged or undirected GNN
        attention_heads: attention heads for transformer
            and for GNN if using e.g. GATConv operator
        feedforward_dim: dim for the transformer MLP
        num_blocks: #transformer blocks
        activation: transformer activation {"gelu", "relu"}
        mlp_dim: classifier hiddden dim
        mlp_num_layers
        agg_mask_output: whether to project each sequence output to
            the vocabulary and output the aggregated logits additionally
        token_logit_output: additionally compute and output logits for each
            individual token
        decoder_network: use a decoder network, which does attend to
            all the graph embeddings, DEPRECATED
        convolution_operator: GNN operator to use
        data_pd: patient records from MIMIC for co-occurrence graph,
            only relevant if using heterogenuous_graph_module
        co_occurrence: co-occurrence node type, one of {visit, patient},
            only relevant if using heterogenuous_graph_module
        co_occurrence_subsample: subsampling ratio [0, 1] for co-occurrence nodes
        co_occurrence_loss: alpha parameter for the additional co-occurrence
            node autoencoder loss
        co_occurrence_dropout: dropout to apply on the edges towards the co-nodes
        co_occurrence_divisor: Divisor to reduce occurrence node dimension to save memory
        co_latent_space: the final visit embedding becomes a convex combination of patient
            node embeddings from the graph by attention of the visit CLS token
        co_occurrence_cluster: reduce num co nodes by clustering
        co_occurrence_features: path to DataFrame with stored static node features
        co_occurrence_batch_size: training batch size for the co-occurrence loss
        triplet_loss: alpha parameter for the additional triplet loss
        triplet_batch_size: -
        triplet_margin: margin parameter of the triplet loss
        graph_memory_size: size of the memory compressing the co occurrence nodes
            if 0, no memory will be used, if memory is used it will be attended to
            and extracted features used as a additional feature for the visit encoding
        umls_graph: Dict
            UMLS CUI graph data
        co_link_config: CoLinkConfig
            config object for co occurrence links
        with_text: bool
            additional text token input
        contractions_type: str
            perform graph contractions i.e. edge pooling with provided score func
        trainable_edge_weights: bool
            make edge weights trainable
        cache_graph_forward: bool
            -
        text_targets: bool
            additionally predict the encoded text tokens
        """
        super(JointCodePretrainingTransformer, self).__init__()

        assert not decoder_network, "`decoder_network` argument is deprecated"
        assert not graph_staged, "`graph_staged` argument is deprecated"
        assert not co_latent_space, "`co_latent_space` argument is deprecated"
        if graph_memory_size > 0:
            assert co_occurrence is not None, "memory can only be used with co-occurrence nodes"

        disease_vocabulary = tokenizer.disease_vocabulary
        prescription_vocabulary = tokenizer.prescription_vocabulary

        self.padding_id = padding_id
        self.cls_id = cls_id
        self.with_text = with_text
        self.text_targets = text_targets
        logging.info(f"[NN] Training with text: {self.with_text}, targets: {self.text_targets}")

        self.attention_heads = attention_heads
        self.cache_graph_forward = cache_graph_forward
        logging.info(f"[NN] cache graph forward pass: {self.cache_graph_forward}")

        # (Graph) embeddings
        if graph_num_layers == 0:
            logging.info("[NN] normal embedding module, no graph")
            embedding_size = (
                num_special_tokens
                + len(disease_vocabulary.idx2word)
                + len(prescription_vocabulary.idx2word)
            )
            self.embedding = FlatEmbedding(embedding_size, embedding_dim)  # type: ignore

        elif umls_graph is not None:
            logging.info("[NN] UMLS graph embedding module")
            self.embedding = UMLSGraphEmbedding(  # type: ignore
                umls_data=umls_graph,
                disease_vocabulary=disease_vocabulary,
                prescription_vocabulary=prescription_vocabulary,
                embedding_dim=embedding_dim,
                graph_num_layers=graph_num_layers,
                graph_num_filters=graph_num_filters,
                convolution_operator=convolution_operator,
                attention_heads=attention_heads,
                num_special_tokens=num_special_tokens,
                data_pd=data_pd,
                co_occurrence=co_occurrence,
                co_occurrence_subsample=co_occurrence_subsample,
                co_occurrence_loss=co_occurrence_loss,
                tokenizer=tokenizer,
                co_occurrence_divisor=1,
                co_occurrence_dropout=co_occurrence_dropout,
                co_occurrence_features=co_occurrence_features,
                co_occurrence_batch_size=co_occurrence_batch_size,
                triplet_loss=triplet_loss,
                triplet_margin=triplet_margin,
                triplet_batch_size=triplet_batch_size,
                co_link_config=co_link_config,
                contractions_type=contractions_type,
            )
            self.co_occurrence = co_occurrence

        else:
            logging.info("[NN] heterogenuous graph embedding module")
            self.embedding = HeterogenousOntologyEmbedding(  # type: ignore
                disease_vocabulary=disease_vocabulary,
                prescription_vocabulary=prescription_vocabulary,
                embedding_dim=embedding_dim,
                graph_num_layers=graph_num_layers,
                graph_num_filters=graph_num_filters,
                convolution_operator=convolution_operator,
                attention_heads=attention_heads,
                num_special_tokens=num_special_tokens,
                data_pd=data_pd,
                co_occurrence=co_occurrence,
                co_occurrence_subsample=co_occurrence_subsample,
                co_occurrence_loss=co_occurrence_loss,
                tokenizer=tokenizer,
                divisor=co_occurrence_divisor,
                co_occurrence_dropout=co_occurrence_dropout,
                co_occurrence_cluster=co_occurrence_cluster,
                co_occurrence_features=co_occurrence_features,
                co_occurrence_batch_size=co_occurrence_batch_size,
                triplet_loss=triplet_loss,
                triplet_margin=triplet_margin,
                triplet_batch_size=triplet_batch_size,
                co_link_config=co_link_config,
                contractions_type=contractions_type,
                trainable_edge_weights=trainable_edge_weights,
            )
            self.co_occurrence = co_occurrence

        # to handle negation feature
        if self.with_text:
            embedding_dim += 1

        # Transformer interaction modeling
        self.decoder_network = decoder_network
        self.encoder = TransformerEncoder(
            hidden_dim=embedding_dim,
            attention_heads=attention_heads,
            feedforward_dim=feedforward_dim,
            activation=activation,
            num_blocks=num_blocks,
        )

        # compress co-occurrence nodes and extract
        # information from compressed nodes
        self.graph_memory = False
        if graph_memory_size > 0:
            self.graph_memory = True
            self.graph_memory_size = graph_memory_size

            assert_msg = f"Embedding needs to be of type {HeterogenousOntologyEmbedding}, but is {type(self.embedding)}"
            assert isinstance(self.embedding, HeterogenousOntologyEmbedding), assert_msg

            logging.info(f"[NN] init compression memory, size: {graph_memory_size}")
            self.co_memory = AsymmetricMemoryAttention(graph_memory_size, embedding_dim)

        # Heads for classification and signal generation
        classifier_input_dim = embedding_dim * 2 if self.graph_memory else embedding_dim
        if self.with_text:
            classifier_input_dim += embedding_dim

        self.disease_classifiers = nn.ModuleList(
            [
                VisitPredictionHead(
                    len(disease_vocabulary.word2idx),
                    input_dim=classifier_input_dim,
                    mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
                ),
                VisitPredictionHead(
                    len(disease_vocabulary.word2idx),
                    input_dim=classifier_input_dim,
                    mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
                ),
            ]
        )

        self.prescr_classifiers = nn.ModuleList(
            [
                VisitPredictionHead(
                    len(prescription_vocabulary.word2idx),
                    input_dim=classifier_input_dim,
                    mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
                ),
                VisitPredictionHead(
                    len(prescription_vocabulary.word2idx),
                    input_dim=classifier_input_dim,
                    mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
                ),
            ]
        )

        self.agg_mask_output = agg_mask_output
        self.token_logit_output = token_logit_output
        assert not (
            self.agg_mask_output and self.token_logit_output
        ), "Only support one additional output"

        # for aggregated output on the tokens
        # for additional Sum-Loss (mentioned in the paper)
        if agg_mask_output:
            logging.info("[NN] Initializing heads for aggregation loss")

            self.disease_mask_classifier = MaskedAggregatorPredictionHead(
                len(disease_vocabulary.word2idx),
                input_dim=embedding_dim,
                mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
            )

            self.prescr_mask_classifier = MaskedAggregatorPredictionHead(
                len(prescription_vocabulary.word2idx),
                input_dim=embedding_dim,
                mlp_hidden_dims=list(repeat(mlp_dim, mlp_num_layers)),
            )

        # for set matching loss (Hungarian set Loss)
        elif token_logit_output:
            logging.info("[NN] Initializing MLPs for set matching loss")
            mlp_hidden_dims = list(repeat(mlp_dim, mlp_num_layers))

            dims = [embedding_dim] + mlp_hidden_dims + [len(disease_vocabulary.word2idx)]
            self.disease_token_clf = MLP(dims)

            dims = [embedding_dim] + mlp_hidden_dims + [len(prescription_vocabulary.word2idx)]
            self.prescription_token_clf = MLP(dims)

        # Text Target prediction head
        if self.text_targets:
            umls_vocabulary = umls_graph["tokenizer"].vocabulary  # type: ignore
            dims = [embedding_dim, len(umls_vocabulary.word2idx)]
            self.text_token_clf = MLP(dims)


    def forward(self, x: Tuple[torch.Tensor, ...], fully_cache_embeddings: bool = False):

        # get data tensors
        disease_tokens = x[0]
        prescr_tokens = x[1]
        seq_length = disease_tokens.shape[1]
        if hasattr(self, "with_text") and self.with_text:
            text_tokens = x[2][:, 0]  # first dimension actual text tokens
            negation_mask = x[2][:, 1]  # second dim. negation mask

        # retrieve graph embeddings
        disease_emb = self.embedding(disease_tokens, use_cache=fully_cache_embeddings)
        prescr_emb = self.embedding(
            prescr_tokens,
            use_cache=(hasattr(self, "cache_graph_forward") and self.cache_graph_forward),
        )
        if hasattr(self, "with_text") and self.with_text:
            text_emb = self.embedding(text_tokens, use_cache=self.cache_graph_forward)

        # transformer interaction modeling
        disease_mask_pad = disease_tokens == self.padding_id
        prescr_mask_pad = prescr_tokens == self.padding_id
        if hasattr(self, "with_text") and self.with_text:
            text_mask_pad = text_tokens == self.padding_id

        # compute CLS masks
        disease_mask_cls = disease_tokens == self.cls_id
        prescr_mask_cls = prescr_tokens == self.cls_id

        # compute joint PAD and CLS mask
        disease_mask_full = torch.logical_or(disease_mask_pad, disease_mask_cls)
        prescr_mask_full = torch.logical_or(prescr_mask_pad, prescr_mask_cls)

        # compute attention masks
        num_attention_heads = self.encoder.encoder.layers[0].self_attn.num_heads
        disease_mask_att = set_first_mask_entry(disease_mask_pad)  # type: ignore
        prescr_mask_att = set_first_mask_entry(prescr_mask_pad)  # type: ignore
        if hasattr(self, "with_text") and self.with_text:
            text_mask_att = set_first_mask_entry(text_mask_pad)  # type: ignore

        if hasattr(self, "with_text") and self.with_text:

            disease_negation_full = torch.logical_not(disease_mask_full)
            prescr_negation_full = torch.logical_not(prescr_mask_full)

            disease_emb = torch.cat((disease_emb, disease_negation_full.unsqueeze(-1)), dim=-1)
            prescr_emb = torch.cat((prescr_emb, prescr_negation_full.unsqueeze(-1)), dim=-1)

            # print(d_pad.shape, d_pad)
            disease_encoded = self.encoder(disease_emb, mask=disease_mask_att)
            prescr_encoded = self.encoder(prescr_emb, mask=prescr_mask_att)

            # approach with concatenated negation indicator
            text_emb = torch.cat((text_emb, negation_mask.unsqueeze(-1)), dim=-1)

            text_encoded = self.encoder(text_emb, mask=text_mask_att)

        else:
            disease_encoded = self.encoder(disease_emb, mask=disease_mask_att)
            prescr_encoded = self.encoder(prescr_emb, mask=prescr_mask_att)

        # *_encoded: (BATCH, CODES, EMBEDDING)
        visit_encoding_d = disease_encoded[:, 0]
        visit_encoding_p = prescr_encoded[:, 0]
        if hasattr(self, "with_text") and self.with_text:
            visit_encoding_text = text_encoded[:, 0]  # get CLS
            visit_encoding_d = torch.cat((visit_encoding_d, visit_encoding_text), dim=1)
            visit_encoding_p = torch.cat((visit_encoding_p, visit_encoding_text), dim=1)
        # visit_encoding_*: (BATCH, EMBEDDING)

        # get additional information from the compressed memory
        # of the co-occurrence nodes
        if self.graph_memory:

            # get graph nodes: (N_CO, EMBEDDING),
            co_nodes = self.embedding.get_all_graph_embeddings()[self.co_occurrence]  # type: ignore

            # extract from compressed memory
            co_memory_d = self.co_memory(visit_encoding_d, co_nodes)  # (BATCH, EMBEDDING)
            co_memory_p = self.co_memory(visit_encoding_p, co_nodes)  # (BATCH, EMBEDDING)

            # concat extracted feature vectors
            visit_encoding_d = torch.cat([visit_encoding_d, co_memory_d], dim=1)
            visit_encoding_p = torch.cat([visit_encoding_p, co_memory_p], dim=1)
            # visit_encoding_*: (BATCH, EMBEDDING * 2)

        # classify [CLS]
        # predict diseases
        logits_d2d = self.disease_classifiers[0](visit_encoding_d)
        logits_p2d = self.disease_classifiers[1](visit_encoding_p)

        # predict medications
        logits_p2p = self.prescr_classifiers[0](visit_encoding_p)
        logits_d2p = self.prescr_classifiers[1](visit_encoding_d)

        # classify and aggregate non-masked tokens (all but CLS and PAD)
        if self.agg_mask_output:

            disease_logits_masked = self.disease_mask_classifier(
                disease_encoded, mask=disease_mask_full
            )
            prescr_logits_masked = self.prescr_mask_classifier(
                prescr_encoded, mask=prescr_mask_full
            )

            logits_masked = (
                disease_logits_masked,
                prescr_logits_masked,
            )

            if hasattr(self, "text_targets") and self.text_targets:
                logits_text = self.text_token_clf(visit_encoding_text)
                logits_masked = (logits_masked, logits_text)

            return (logits_d2d, logits_p2d, logits_p2p, logits_d2p), logits_masked

        elif self.token_logit_output:

            disease_logits = self.disease_token_clf(disease_encoded)
            prescr_logits = self.prescription_token_clf(prescr_encoded)

            clf_logits = (logits_d2d, logits_p2d, logits_p2p, logits_d2p)
            token_data = (
                disease_logits,
                prescr_logits,
                disease_mask_full,
                prescr_mask_full,
            )

            return clf_logits, token_data

        return (logits_d2d, logits_p2d, logits_p2p, logits_d2p), (None, None)
