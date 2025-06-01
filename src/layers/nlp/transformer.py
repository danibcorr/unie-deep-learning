# 3pps
import torch
from torch import nn
from torch.nn import functional as F

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
		# Input_tensor (B, ...) -> (B, ..., d_model)
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
		# (B, ..., d_model) -> (B, sequence_length, d_model)
		# Seleccionamos
		x = input_embedding + (
			self.pe_matrix[:, : input_embedding.shape[1], :]
		).requires_grad_(False)
		return self.dropout(x)
	
class LayerNormalization(nn.Module):

	def __init__(self, eps: float = 1e-6) -> None:

		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self.eps = eps

		# Utilizamos un factor alpha para multiplicar el valor de la normalización
		self.alpha = nn.Parameter(torch.ones(1))
		# Utilizamos un factor del sesgo para sumar
		self.bias = nn.Parameter(torch.zeros(1))

	def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
		# (B, sequence_length, d_model)
		mean = torch.mean(input=input_embedding, dim=-1, keepdim=True)
		var = torch.var(input=input_embedding, dim=-1, keepdim=True)
		return self.alpha * ((input_embedding - mean) / (torch.sqrt(var + self.eps))) + self.bias
	

class FeedForward(nn.Module):

	def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
		
		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self.d_model = d_model
		self.d_ff = d_ff

		# Creamos el modelo secuencial
		self.ffn = nn.Sequential(
			nn.Linear(in_features=self.d_model, out_features=self.d_ff),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(in_features=self.d_ff, out_features=self.d_model),
		)


	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		# (B, sequence_length, d_model)
		return self.ffn(input_tensor)
	

class MultiHeadAttention(nn.Module):

	def __init__(self, d_model: int, h: int, dropout_rate: float, mask: bool) -> None:

		"""_summary_
		h: numero de cabezas.
		"""

		super().__init__()

		assert self.d_model % h == 0, "d_model ha de ser divisible entre h"
		
		self.d_model = d_model
		self.h = h		
		self.d_k = self.d_model // self.h
		self.d_v = self.d_v // self.h
		self.mask = mask

		# Parámetros
		self.W_K = nn.Linear(in_features=self.d_model, out_features=self.d_model)
		self.W_Q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
		self.W_V = nn.Linear(in_features=self.d_model, out_features=self.d_model)

		self.dropout = nn.Dropout(dropout_rate)

	def scaled_dot_product(self, k: torch.Tensor,  q: torch.Tensor, v: torch.Tensor, mask: bool):

		# Primero realizamos el producto matricial
		matmul_q_k = (q @ k.transpose(-2, -1))
		
		# Luego realizamos el escalado
		matmul_q_k_scaled = matmul_q_k / torch.sqrt(self.d_k)

		# El enmascarado es para el decoder, relleno de infinitos
		if mask:
			# Creamos una matriz de ceros igual que el producto matricial y
			# Obtener los índices de la diagonal superior estricta con infinitos
			mask_matrix = torch.triu(torch.full_like(matmul_q_k_scaled, float('-inf')), diagonal=1)

			# Sumamos la mascara
			matmul_q_k_scaled += mask_matrix

		return F.softmax(matmul_q_k_scaled) @ v

	def forward(self, k: torch.Tensor,  q: torch.Tensor, v: torch.Tensor, mask: bool) -> torch.Tensor:

		# k -> (Batch, seq_len, d_model) igual para el resto
		key_prima = self.W_K(k)
		query_prima = self.W_Q(q)
		value_prima = self.W_V(v)

		# Cambiamos las dimensiones y hacemos el split de los embedding para cada head
		# Pasando de (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
		key_prima = key_prima.view(key_prima.shape[0], key_prima.shape[1], self.h, self.d_k).transpose(1, 2)
		query_prima = query_prima.view(query_prima.shape[0], query_prima.shape[1], self.h, self.d_k).transpose(1, 2)
		value_prima = value_prima.view(value_prima.shape[0], value_prima.shape[1], self.h, self.d_k).transpose(1, 2)
