import torch
from torch import nn


class SqueezeExcitation(nn.Module):
	def __init__(self, channel_size: int, ratio: int) -> None:
		"""
		Implementación del bloque Squeeze-and-Excitation (SE).

		Args:
			channel_size: Número de canales de entrada
			ratio: Factor de reducción para la capa de compresión
		"""

		# Constructor de la clase
		super().__init__()

		# Vamos a crear un modelo Sequential
		self.se_block = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),  # (B, C, 1, 1)
			nn.Flatten(),  # (B, C)
			nn.Linear(
				in_features=channel_size, out_features=channel_size // ratio
			),  # (B, C//ratio)
			nn.ReLU(),  # (B, C//ratio)
			nn.Linear(
				in_features=channel_size // ratio, out_features=channel_size
			),  # (B, C)
			nn.Sigmoid(),
		)

		# Aunque también lo podemos hacer por capa
		# self.pooling = nn.AdaptiveAvgPool2d((1, 1))
		# self.linear_1 = nn.Linear(channel_size, channel_size//ratio)
		# self.linear_2 = nn.Linear(channel_size//ratio, channel_size)
		# self.act_relu = nn.ReLU()
		# self.act_sigmoid = nn.Sigmoid()

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		# x = self.pooling(input_tensor)
		# x = x.squeeze()
		# print(x.shape)

		# x = self.linear_1(x)
		# print(x.shape)

		# x = self.act_relu(x)
		# print(x.shape)

		# x = self.linear_2(x)
		# print(x.shape)

		# x = self.act_sigmoid(x)
		# x = torch.unsqueeze(x, -1)
		# x = torch.unsqueeze(x, -1)
		# print(x.shape)

		# Primero podemos obtener el tamaño del tensor de entrada
		b, c, _, _ = input_tensor.size()

		# Obtenemos el tensor de aplicar SE
		x = self.se_block(input_tensor)

		# Modificamos el shape del tensor para ajustarlo al input
		x = x.view(b, c, 1, 1)

		# Aplicamos el producto como mecanismo de atención
		return x * input_tensor
