# 3pps
import torch


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Creates a causal mask to prevent the decoder from attending
    to future tokens during training.

    Args:
        size: Length of the sequence.

    Returns:
        Causal mask of shape (size, size).
    """

    return torch.tril(torch.ones(size, size))


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Creates a mask to ignore padding tokens in a sequence.

    Args:
        seq: Sequence of tokens, shape (B, seq_len).
        pad_token: Padding token value.

    Returns:
        Padding mask of shape (B, 1, 1, seq_len).
    """

    return (seq != pad_token).unsqueeze(1).unsqueeze(1)
