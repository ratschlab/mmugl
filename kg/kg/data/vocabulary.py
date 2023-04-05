# ===============================================
#
# Vocabulary/Tokenizer class and utilities
#
# ===============================================
import logging
import random
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd
import torch


# ===================================
#
# Vocabulary
#
# ===================================
class Vocabulary(object):
    """
    Helper class to hold word/code to id mappings

    Attributes
    ----------
    idx2word: dictionary: ids -> words
    word2idx: dictionary: words -> ids
    """

    def __init__(self):
        self.idx2word = OrderedDict()
        self.word2idx = OrderedDict()

    def add_sentence(self, sentence: Union[Set[Any], Sequence[Any]]):
        """
        Add a sequence of tokens to the `Vocabulary`
        """
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)

    def __str__(self) -> str:
        return f"`{self.__class__.__name__}`: {len(self.idx2word)} words"

    def __repr__(self) -> str:
        return str(self)


# ===================================
#
# Tokenizer
#
# ===================================
class SimpleTokenizer(object):
    """
    Tokenizes a vocabulary

    Attributes
    ----------
    vocabulary: `Vocabulary`
    """

    def __init__(
        self,
        codes: Union[Set[str], Sequence[str]],
        special_tokens: Sequence[str] = ("[PAD]", "[CLS]", "[MASK]"),
    ):
        """
        Constructor for `DiagnosisTokenizer

        Parameters
        ----------
        codes: list of codes in the vocabulary
        special_tokens: collection of special tokens
        """
        self.vocabulary = Vocabulary()

        # initialize vocabulary
        self.vocabulary.add_sentence(special_tokens)
        self.vocabulary.add_sentence(codes)

    def convert_tokens_to_ids(self, tokens: Sequence[Any]) -> Sequence[int]:
        """Converts a sequence of tokens into ids using the vocabulary"""
        return list(map(lambda token: self.vocabulary.word2idx[token], tokens))

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> Sequence[Any]:
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        return list(map(lambda id: self.vocabulary.idx2word[id], ids))


class DiagnosisTokenizer(object):
    """
    Tokenizes a dataset of diagnosis codes

    Attributes
    ----------
    vocabulary: `Vocabulary`
    disease_vocabulary: `Vocabulary` containing only the ICD codes
        for this Tokenizer

    """

    def __init__(
        self,
        diagnosis_codes: Union[Set[str], Sequence[str]],
        special_tokens: Sequence[str] = ("[PAD]", "[CLS]", "[MASK]"),
    ):
        """
        Constructor for `DiagnosisTokenizer

        Parameters
        ----------
        diagnosis_codes: list of codes in the vocabulary
        special_tokens: collection of special tokens
        """

        self.vocabulary = Vocabulary()
        self.disease_vocabulary = Vocabulary()

        # initialize vocabulary
        self.vocabulary.add_sentence(special_tokens)
        self.vocabulary.add_sentence(diagnosis_codes)
        self.disease_vocabulary.add_sentence(diagnosis_codes)

    def convert_tokens_to_ids(self, tokens: Sequence[Any]) -> Sequence[int]:
        """Converts a sequence of tokens into ids using the vocabulary"""
        return list(map(lambda token: self.vocabulary.word2idx[token], tokens))

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> Sequence[Any]:
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        return list(map(lambda id: self.vocabulary.idx2word[id], ids))


class CodeTokenizer(DiagnosisTokenizer):
    """
    Tokenize a dataset of diagnosis (ICD) and prescription (ATC) codes

    Attributes
    ----------
    vocabulary: `Vocabulary`
    disease_vocabulary: `Vocabulary` containing only the ICD codes
        for this Tokenizer
    prescription_vocabulary: `Vocabulary containing only the ATC codes
        for this Tokenizer
    """

    def __init__(
        self,
        diagnosis_codes: Union[Set[str], Sequence[str]],
        prescription_codes: Union[Set[str], Sequence[str]],
        special_tokens: Sequence[str] = ("[PAD]", "[CLS]", "[MASK]"),
    ):
        """
        Constructor for `DiagnosisTokenizer

        Parameters
        ----------
        diagnosis_codes: list of codes in the vocabulary
        special_tokens: collection of special tokens
        """
        super().__init__(diagnosis_codes, special_tokens)

        # additional prescription vocabulary
        self.prescription_vocabulary = Vocabulary()

        # initialize vocabulary
        self.vocabulary.add_sentence(prescription_codes)
        self.prescription_vocabulary.add_sentence(prescription_codes)
