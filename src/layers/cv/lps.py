"""
Este clase implementa la capa APS de este paper: https://arxiv.org/abs/2210.08001
"""

# Standard libraries

# 3pps
import torch
from torch import nn
from torch.nn import functional as F


class LPS(nn.Module):
	def __init__(self, channel_size: int, hidden_size: int) -> None:
		"""
		Initializes the model with specified channel and hidden sizes.

		Args:
			channel_size: Number of input channels for the Conv2D layer.
			hidden_size: Number of hidden units for the Conv2D layer.
		"""

		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self._stride = 2

		# Definimos el modelo único para cada componente
		self.conv_model = nn.Sequential(
			nn.Conv2d(
				in_channels=channel_size,
				out_channels=hidden_size,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=hidden_size,
				out_channels=hidden_size,
				kernel_size=3,
				stride=1,
				padding=1,
			),
			nn.Flatten(),
			nn.AdaptiveAvgPool2d(1),
		)

	def forward(
		self, input_tensor: torch.Tensor, return_index: bool = False
	) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		"""
		Processes input to extract dominant polyphase component.

		Args:
			input_tensor: Tensor with shape (B, C, H, W).
			return_index: If True, returns index of dominant component.

		Returns:
			Tensor of dominant component, optionally with index.
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

		# Utilizamos el modelo basado en convoluciones por cada componente
		_logits = []
		for polyphase in range(polyphase_combined.size()[1]):
			_logits.append(self.conv_model(polyphase_combined[:, polyphase, ...]))
		logits = torch.squeeze(torch.stack(_logits))

		# Aplicamos la norma a la última dimensión
		polyphase_norms = F.gumbel_softmax(logits, tau=1, hard=False)

		# Seleccionamos el componente polifásico de mayor orden
		polyphase_max_norm = torch.argmax(polyphase_norms)

		# Obtenemos el componente polifásico de mayor orden
		output_tensor = polyphase_combined[:, polyphase_max_norm, ...]

		# En el paper existe la opción de devolver el índice
		if return_index:
			return output_tensor, polyphase_max_norm

		# En caso contrario solo devolvemos el tensor
		return output_tensor
