# 3pps
import torch
from torch import nn


class InputEmbedding(nn.Module):
	def __init__(self, d_model: int, vocab_size: int) -> None:
		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self.d_model = d_model
		self.vocab_size = vocab_size

		# Utilizamos la capa Embedding de PyTorch que funciona como
		# una tabal lookup that stores embeddings of a fixed dictionary and size.
		# Osea que es un diccionario que tiene por cada token, hasta un total de
		# vocab_size, un vector de tamaño d_model. En el paper: we use learned
		# embeddings to convert the input tokens and output tokens to vectors
		# of dimension dmodel
		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		# Paper: In the embedding layers, we multiply those weights by sqrt(d_model)
		return self.embedding(input_tensor) * torch.sqrt(self.d_model)


class PossitionalEncoding(nn.Module):
	def __init__(self, d_model: int, sequence_length: int, dropout_rate: float) -> None:
		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self.d_model = d_model

		# Cuando le damos una secuencia de tokens, tenemos que saber
		# la longitud máxima de la secuencia
		self.sequence_length = sequence_length
		self.dropout = nn.Dropout(dropout_rate)

		# Creamos una matriz del positional embedding
		# (sequence_length, d_model)
		pe_matrix = torch.zeros(size=(self.sequence_length, self.d_model))

		# Ahora rellenamos la matriz de posiciones
		# La posición va hasta el máximo de la longitud de la secuencia
		for pos in range(self.sequence_length):
			for i in range(0, d_model, 2):
				# Para las posiciones pares usamos el seno
				pe_matrix[pos, i] = torch.sin(pos / (10000 ** ((2 * i) / d_model)))
				# Para las posiciones impares usamos el coseno
				pe_matrix[pos, i + 1] = torch.cos(
					pos / (10000 ** ((2 * (i + 1)) / d_model))
				)

		# Tenemos que convertirlo a (1, sequence_length, d_model) para
		# procesarlo por lotes
		pe_matrix = pe_matrix.unsqueeze(0)

		# Esta matriz no se aprende, es fija, la tenemos que guardar con el modelo
		self.register_buffer(name="pe_matrix", tensor=pe_matrix)

	def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
		# Seleccionamos
		x = input_embedding + (
			self.pe_matrix[:, : input_embedding.shape[1], :]
		).requires_grad_(False)
		return self.dropout(x)
