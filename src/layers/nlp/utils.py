# 3pps
import torch


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Crea una máscara causal (triangular inferior) para prevenir que el decoder
    vea tokens futuros durante el entrenamiento.

    Args:
            size: Tamaño de la secuencia

    Returns:
            Máscara causal de tamaño (size, size)
    """

    return torch.tril(torch.ones(size, size))


def create_padding_mask(seq: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    Crea una máscara para ignorar tokens de padding.

    Args:
            seq: Secuencia de tokens (B, seq_len)
            pad_token: Valor del token de padding

    Returns:
            Máscara de padding (B, 1, 1, seq_len)
    """

    return (seq != pad_token).unsqueeze(1).unsqueeze(1)
