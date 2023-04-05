# ===============================================
#
# Loss functions
#
# ===============================================
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


def hungarian_ce_loss(
    x: torch.Tensor, y: torch.Tensor, x_mask=None, verbose: bool = False
) -> torch.Tensor:
    """
    CE loss for sets loosely based on the implementation of the paper:
    `Object-Centric Learning with Slot Attention` found here:
    https://github.com/google-research/google-research/blob/master/slot_attention/utils.py

    Computes the pairwise CE loss and then tries to optimize a matching
    between the predictions `x` and the targets `y` using the Hungarian algorithm

    Will compute the pairwise loss between all the predicted targets and
    at most `seq_dim` (i.e. maximum sequence length) many targets to
    reduce computational load. If more targets, the first #`seq_dim` will
    be considered

    Parameter
    ---------
    x: predictions, shape [batch, seq, vocab]
    y: targets, shape [batch, vocab] (multi-label)
    x_mask: x predictions to mask (masks where True), shape [batch, seq]
    verbose: -

    Returns
    -------
    Aggregated loss as torch tensor
    """

    IGNORE_INDEX = -100
    LARGE_VALUE = 1e6

    batch_dim = x.shape[0]
    seq_dim = x.shape[1]
    vocab_dim = x.shape[2]

    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

    # expand x for pairwise computation
    x_expanded = x.unsqueeze(-3).expand(-1, seq_dim, -1, -1).reshape(batch_dim, -1, vocab_dim)
    if verbose:
        print(f"x expanded: {x_expanded.shape}")

    # compute class indeces and expand for pairwise loss computation
    y_indeces = torch.where(
        y == 1,
        torch.arange(0, vocab_dim, 1, device=device, dtype=torch.float32),
        torch.tensor([IGNORE_INDEX], device=device).float(),
    )

    # TODO: make this part faster with proper
    # gather or similar operations
    def extract_targets(sample):
        """Extract class targets"""

        # extract relevant class targets
        sample = sample[sample != IGNORE_INDEX]

        # collect them
        result = torch.empty((seq_dim,), device=device, dtype=torch.float32)
        result[0:seq_dim] = IGNORE_INDEX
        extract_length = min(len(sample), len(result))
        result[0:extract_length] = sample[0:extract_length]

        return result

    y_indeces_select = torch.stack([extract_targets(b) for b in torch.unbind(y_indeces)])

    # expand indeces for all-to-all pairwise computation
    y_index_exp = y_indeces_select.unsqueeze(-1).expand(-1, -1, seq_dim).reshape(batch_dim, -1)
    if verbose:
        print(f"y index exp: {y_index_exp.shape}")

    # compute pairwise loss
    pairwise_loss = F.cross_entropy(
        x_expanded.view(-1, vocab_dim),
        y_index_exp.view(-1).long(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(batch_dim, -1)

    # replace distance with Inf where no target label
    pairwise_loss = torch.where(
        y_index_exp == IGNORE_INDEX,
        torch.tensor([LARGE_VALUE], device=device),
        pairwise_loss,
    )

    # expand and apply mask on x (mask CLS and PAD)
    mask_expanded = x_mask.unsqueeze(-2).expand(-1, seq_dim, -1)
    pairwise_loss = torch.where(
        mask_expanded,
        torch.tensor([LARGE_VALUE], device=device),
        pairwise_loss.view(batch_dim, seq_dim, seq_dim),
    )

    # move loss results to cpu
    pairwise_cpu = pairwise_loss.detach().cpu()

    # optimize matching using hungarian algorithm
    indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cpu)))
    # if verbose: print(f"Optim indeces: {indices}")

    # collected according to computed best matches
    collected = torch.stack(
        [b[row_ind, col_ind] for b, (row_ind, col_ind) in zip(torch.unbind(pairwise_loss), indices)]
    )

    collected = torch.where(collected == LARGE_VALUE, torch.tensor([0.0], device=device), collected)

    # aggregate
    mean_mask = collected != 0
    loss = (collected * mean_mask).sum() / (mean_mask.sum())

    return loss


# ===== DEPRECATED =====
def hungarian_ce_loss_full_vocab(
    x: torch.Tensor, y: torch.Tensor, x_mask=None, verbose: bool = False
) -> torch.Tensor:
    """
    CE loss for sets based on the implementation of the paper:
    `Object-Centric Learning with Slot Attention` found here:
    https://github.com/google-research/google-research/blob/master/slot_attention/utils.py

    Computes the pairwise CE loss and then tries to optimize a matching
    between the predictions `x` and the targets `y` using the Hungarian algorithm

    Parameter
    ---------
    x: predictions, shape [batch, seq, vocab]
    y: targets, shape [batch, vocab] (multi-label)
    x_mask: x predictions to mask (masks where True), shape [batch, seq]
    verbose: -

    Returns
    -------
    Aggregated loss as torch tensor
    """

    IGNORE_INDEX = -100
    LARGE_VALUE = 1e6

    batch_dim = x.shape[0]
    seq_dim = x.shape[1]
    vocab_dim = x.shape[2]

    device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

    # expand x for pairwise computation
    x_expanded = x.unsqueeze(-3).expand(-1, vocab_dim, -1, -1).reshape(batch_dim, -1, vocab_dim)
    if verbose:
        print(f"x expanded: {x_expanded.shape}")

    # compute class indeces and expand for pairwise loss computation
    y_indeces = torch.where(
        y == 1,
        torch.arange(0, vocab_dim, 1, device=device, dtype=torch.float32),
        torch.tensor([IGNORE_INDEX], device=device).float(),
    )
    y_index_exp = y_indeces.unsqueeze(-1).expand(-1, -1, seq_dim).reshape(batch_dim, -1)
    if verbose:
        print(f"y index exp: {y_index_exp.shape}")

    # compute pairwise loss
    pairwise_loss = F.cross_entropy(
        x_expanded.view(-1, vocab_dim),
        y_index_exp.view(-1).long(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(batch_dim, -1)

    # replace distance with Inf where no target label
    pairwise_loss = torch.where(
        y_index_exp == IGNORE_INDEX,
        torch.tensor([LARGE_VALUE], device=device),
        pairwise_loss,
    )

    # expand and apply mask on x (mask CLS and PAD)
    mask_expanded = x_mask.unsqueeze(-2).expand(-1, vocab_dim, -1)
    pairwise_loss = torch.where(
        mask_expanded,
        torch.tensor([LARGE_VALUE], device=device),
        pairwise_loss.view(batch_dim, vocab_dim, seq_dim),
    )

    # move loss results to cpu
    pairwise_cpu = pairwise_loss.detach().cpu()

    if verbose:
        print(f"pairwise loss: {pairwise_cpu.shape}")
        print(pairwise_cpu)

    # optimize matching using hungarian algorithm
    indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cpu)))
    if verbose:
        print(f"Optim indeces: {indices}")

    collected = torch.stack(
        [b[row_ind, col_ind] for b, (row_ind, col_ind) in zip(torch.unbind(pairwise_loss), indices)]
    )

    # ignore masking for loss
    if verbose:
        print(f"Collected {collected.shape}")
        print(collected)
    collected = torch.where(collected == LARGE_VALUE, torch.tensor([0.0], device=device), collected)

    # aggregate
    mean_mask = collected != 0
    loss = (collected * mean_mask).sum() / (mean_mask.sum())

    return loss
