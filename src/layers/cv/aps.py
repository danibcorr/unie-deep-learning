"""
Este clase implementa la capa APS de este paper: https://arxiv.org/abs/2011.14214
"""

# Standard libraries
from typing import Literal

# 3pps
import torch
from torch import nn


class APS(nn.Module):
    def __init__(
        self,
        norm: int | float | Literal["fro", "nuc", "inf", "-inf"] | None = 2,
    ) -> None:
        """
        Initializes the class with normalization option.

        Args:
                norm: Normalization type or value, defaults to 2.
        """

        # Constructor de la clase
        super().__init__()

        # Definimos los parámetros de la clase
        self._stride = 2
        self.norm = norm

    def forward(
        self, input_tensor: torch.Tensor, return_index: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Processes input tensor to extract dominant polyphase component.

        Args:
                input_tensor: Tensor with shape (B, C, H, W).
                return_index: If True, returns index of dominant component.

        Returns:
                Output tensor, optionally with index if return_index is True.
        """

        # Tenemos a la entrada un tensor de (B, C, H, W)
        # El número de componentes polifásicas coincide con el tamaño
        # de paso elevado al cuadrado, porque nos vemos tanto en la
        # altura como en la anchura , en total 4
        poly_a = input_tensor[:, :, :: self._stride, :: self._stride]
        poly_b = input_tensor[:, :, :: self._stride, 1 :: self._stride]
        poly_c = input_tensor[:, :, 1 :: self._stride, :: self._stride]
        poly_d = input_tensor[:, :, 1 :: self._stride, 1 :: self._stride]

        # Combinamos las componentes en un solo tensor (B, P, C, H, W)
        polyphase_combined = torch.stack((poly_a, poly_b, poly_c, poly_d), dim=1)

        # Extraemos las dimensiones
        b, p, _, _, _ = polyphase_combined.size()

        # Combinamos los valores de los canales, altura y anchura del tensor
        polyphase_combined_reshaped = torch.reshape(polyphase_combined, (b, p, -1))

        # Aplicamos la norma a la última dimensión
        polyphase_norms = torch.linalg.vector_norm(
            input=polyphase_combined_reshaped, ord=self.norm, dim=(-1)
        )

        # Seleccionamos el componente polifásico de mayor orden
        polyphase_max_norm = torch.argmax(polyphase_norms)

        # Obtenemos el componente polifásico de mayor orden
        output_tensor = polyphase_combined[:, polyphase_max_norm, ...]

        # En el paper existe la opción de devolver el índice
        if return_index:
            return output_tensor, polyphase_max_norm

        # En caso contrario solo devolvemos el tensor
        return output_tensor
