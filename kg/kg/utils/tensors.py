# ===============================================
#
# Utilities for manipulating tensors
#
# ===============================================
import torch


def padding_to_attention_mask(padding_mask: torch.BoolTensor, num_heads: int = 1) -> torch.Tensor:
    """
    Transforms a batch of padding masks (BooleanTensors) to an attention
    mask suitable to use with `torch.nn.MultiheadAttention`

    Parameter
    ---------
    padding_mask: torch.BoolTensor
        shape: (Batch, Sequence)
    num_heads: int
        number of attention heads

    Return
    ------
    batch_head_expanded: torch.BoolTensor
        the expanded mask for `MultiheadAttention`
        shape: (Batch * Heads, Sequence, Sequence)
    """

    batch_dim = padding_mask.shape[0]
    sequence_length = padding_mask.shape[1]

    negated_padding_mask = torch.logical_not(padding_mask)
    row_expanded = negated_padding_mask.unsqueeze(-2).expand(-1, sequence_length, -1)
    col_expanded = negated_padding_mask.unsqueeze(-1).expand(-1, -1, sequence_length)

    batch_attention_mask = torch.logical_not(torch.logical_and(row_expanded, col_expanded))
    batch_head_expanded = batch_attention_mask.unsqueeze(1).expand(
        batch_dim, num_heads, sequence_length, sequence_length
    )

    return batch_head_expanded.reshape(-1, sequence_length, sequence_length)


def set_first_mask_entry(batch: torch.BoolTensor, value: bool = False) -> torch.BoolTensor:
    """
    Sets the first entry of a batch of shape (B, S)
    to `value`

    Parameter
    ---------
    batch: torch.BoolTensor
        batch to be adapted
    value: bool
        value to set index 0 to for each
        element in the batch

    Return
    ------
    batch_fixed: torch.BoolTensor
        adjusted batch tensor
    """

    batch_fixed = batch.clone()
    batch_fixed[:, 0] = value

    return batch_fixed  # type: ignore
