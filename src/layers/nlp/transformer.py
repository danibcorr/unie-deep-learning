# Standard libraries
import math

# 3pps
import torch
from torch import nn
from torch.nn import functional as F


class InputEmbedding(nn.Module):
	def __init__(self, d_model: int, vocab_size: int) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			vocab_size (int): _description_
		"""

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
		"""_summary_

		Args:
			input_tensor (torch.Tensor): _description_

		Returns:
			torch.Tensor: _description_
		"""

		# Paper: In the embedding layers, we multiply those weights by sqrt(d_model)
		# Input_tensor (B, ...) -> (B, ..., d_model)
		return self.embedding(input_tensor) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, sequence_length: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			sequence_length (int): _description_
			dropout_rate (float): _description_
		"""

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

		# # Ahora rellenamos la matriz de posiciones
		# # La posición va hasta el máximo de la longitud de la secuencia
		# for pos in range(self.sequence_length):
		# 	for i in range(0, d_model, 2):
		# 		# Para las posiciones pares usamos el seno
		# 		pe_matrix[pos, i] = torch.sin(pos / (10000 ** ((2 * i) / d_model)))
		# 		# Para las posiciones impares usamos el coseno
		# 		pe_matrix[pos, i + 1] = torch.cos(
		# 			pos / (10000 ** ((2 * (i + 1)) / d_model))
		# 		)

		# Crear vector de posiciones
		position = torch.arange(0, self.sequence_length, dtype=torch.float).unsqueeze(1)

		# Crear vector de divisores
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)

		# Aplicar sin y cos
		pe_matrix[:, 0::2] = torch.sin(position * div_term)
		pe_matrix[:, 1::2] = torch.cos(position * div_term)

		# Tenemos que convertirlo a (1, sequence_length, d_model) para
		# procesarlo por lotes
		pe_matrix = pe_matrix.unsqueeze(0)

		# Esta matriz no se aprende, es fija, la tenemos que guardar con el modelo
		self.register_buffer(name="pe_matrix", tensor=pe_matrix)

	def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
		"""_summary_

		Args:
			input_embedding (torch.Tensor): _description_

		Returns:
			torch.Tensor: _description_
		"""

		# (B, ..., d_model) -> (B, sequence_length, d_model)
		# Seleccionamos
		x = input_embedding + (
			self.pe_matrix[:, : input_embedding.shape[1], :]  # type: ignore
		).requires_grad_(False)
		return self.dropout(x)


class LayerNormalization(nn.Module):
	def __init__(self, features: int, eps: float = 1e-6) -> None:
		"""_summary_

		Args:
			features (int): _description_
			eps (float, optional): _description_. Defaults to 1e-6.
		"""

		# Constructor de la clase
		super().__init__()

		# Definimos los parámetros de la clase
		self.features = features
		self.eps = eps

		# Utilizamos un factor alpha para multiplicar el valor de la normalización
		self.alpha = nn.Parameter(torch.ones(self.features))
		# Utilizamos un factor del sesgo para sumar
		self.bias = nn.Parameter(torch.zeros(self.features))

	def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
		"""_summary_

		Args:
			input_embedding (torch.Tensor): _description_

		Returns:
			torch.Tensor: _description_
		"""

		# (B, sequence_length, d_model)
		mean = torch.mean(input=input_embedding, dim=-1, keepdim=True)
		var = torch.var(input=input_embedding, dim=-1, keepdim=True, unbiased=False)
		return (
			self.alpha * ((input_embedding - mean) / (torch.sqrt(var + self.eps)))
			+ self.bias
		)


class FeedForward(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			d_ff (int): _description_
			dropout_rate (float): _description_
		"""

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
		"""_summary_

		Args:
			input_tensor (torch.Tensor): _description_

		Returns:
			torch.Tensor: _description_
		"""

		# (B, sequence_length, d_model)
		return self.ffn(input_tensor)


class MultiHeadAttention(nn.Module):
	def __init__(self, d_model: int, h: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			h (int): _description_
			dropout_rate (float): _description_
		"""

		# Constructor de la clase
		super().__init__()

		# el tamalo de los embeddings debe ser proporcional al número de cabezas
		# para realizar la división, por lo que es el resto ha de ser 0
		assert d_model % h == 0, "d_model ha de ser divisible entre h"

		self.d_model = d_model
		self.h = h
		self.dropout = nn.Dropout(dropout_rate)

		# Valore establecidos en el paper
		self.d_k = self.d_model // self.h
		self.d_v = self.d_model // self.h

		# Parámetros
		self.W_K = nn.Linear(
			in_features=self.d_model, out_features=self.d_model, bias=False
		)
		self.W_Q = nn.Linear(
			in_features=self.d_model, out_features=self.d_model, bias=False
		)
		self.W_V = nn.Linear(
			in_features=self.d_model, out_features=self.d_model, bias=False
		)
		self.W_OUTPUT_CONCAT = nn.Linear(
			in_features=self.d_model, out_features=self.d_model, bias=False
		)

	@staticmethod
	def attention(
		k: torch.Tensor,
		q: torch.Tensor,
		v: torch.Tensor,
		mask: torch.Tensor | None = None,
		dropout: nn.Dropout | None = None,
	):
		"""_summary_

		Args:
			k (torch.Tensor): _description_
			q (torch.Tensor): _description_
			v (torch.Tensor): _description_
			mask (torch.Tensor | None, optional): _description_. Defaults to None.
			dropout (nn.Dropout | None, optional): _description_. Defaults to None.

		Returns:
			_type_: _description_
		"""

		# Primero realizamos el producto matricial con la transpuesta
		# q = (Batch, h, seq_len, d_k)
		# k.T = (Batch, h, d_k, seq_len)
		# matmul_q_k = (Batch, h, seq_len, seq_len)
		matmul_q_k = q @ k.transpose(-2, -1)

		# Luego realizamos el escalado
		d_k = k.shape[-1]
		matmul_q_k_scaled = matmul_q_k / math.sqrt(d_k)

		# El enmascarado es para el decoder, relleno de infinitos
		if mask is not None:
			matmul_q_k_scaled.masked_fill_(mask == 0, -1e9)

		# Obtenemos los scores/puntuación de la atención
		attention_scores = F.softmax(input=matmul_q_k_scaled, dim=-1)

		# Aplicamos dropout
		if dropout is not None:
			attention_scores = dropout(attention_scores)

		# Multiplicamos por el valor
		# attention_scores = (Batch, h, seq_len, seq_len)
		# v = (Batch, h, seq_len, d_k)
		# Output = (Batch, h, seq_len, d_k)
		return (attention_scores @ v), attention_scores

	def forward(
		self,
		k: torch.Tensor,
		q: torch.Tensor,
		v: torch.Tensor,
		mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""_summary_

		Args:
			k (torch.Tensor): _description_
			q (torch.Tensor): _description_
			v (torch.Tensor): _description_
			mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# k -> (Batch, seq_len, d_model) igual para el resto
		key_prima = self.W_K(k)
		query_prima = self.W_Q(q)
		value_prima = self.W_V(v)

		# Cambiamos las dimensiones y hacemos el split de los embedding para cada head
		# Pasando de (Batch, seq_len, d_model) a (Batch, seq_len, h, d_k)
		# Para luego pasar de (Batch, seq_len, h, d_k) a (Batch, h, seq_len, d_k)
		key_prima = key_prima.view(
			key_prima.shape[0], key_prima.shape[1], self.h, self.d_k
		).transpose(1, 2)
		query_prima = query_prima.view(
			query_prima.shape[0], query_prima.shape[1], self.h, self.d_k
		).transpose(1, 2)
		value_prima = value_prima.view(
			value_prima.shape[0], value_prima.shape[1], self.h, self.d_k
		).transpose(1, 2)

		# Obtenemos la matriz de atencion y la puntuación
		# attention = (Batch, h, seq_len, d_k)
		# attention_scores = (Batch, h, seq_len, seq_len)
		attention, attention_scores = MultiHeadAttention.attention(
			k=key_prima,
			q=query_prima,
			v=value_prima,
			mask=mask,
			dropout=self.dropout,
		)

		# Tenemos que concatenar la información de todas las cabezas
		# Queremos (Batch, seq_len, d_model)
		# self.d_k = self.d_model // self.h; d_model = d_k * h
		attention = attention.transpose(1, 2)  # (Batch, seq_len, h, d_k)
		b, seq_len, h, d_k = attention.size()
		# Al parecer, contiguous permite evitar errores de memoria
		attention_concat = attention.contiguous().view(
			b, seq_len, h * d_k
		)  # (Batch, seq_len, h * d_k) = (Batch, seq_len, d_model)

		return self.W_OUTPUT_CONCAT(attention_concat)


class ResidualConnection(nn.Module):
	def __init__(self, features: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			features (int): _description_
			dropout_rate (float): _description_
		"""

		super().__init__()

		self.dropout = nn.Dropout(dropout_rate)
		self.layer_norm = LayerNormalization(features=features)

	def forward(self, input_tensor: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
		"""_summary_

		Args:
			input_tensor (torch.Tensor): _description_
			sublayer (nn.Module): _description_

		Returns:
			torch.Tensor: _description_
		"""

		return input_tensor + self.dropout(sublayer(self.layer_norm(input_tensor)))


class EncoderBlock(nn.Module):
	def __init__(self, d_model: int, d_ff: int, h: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			d_ff (int): _description_
			h (int): _description_
			dropout_rate (float): _description_
		"""

		super().__init__()

		# Parametros
		self.d_model = d_model
		self.d_ff = d_ff
		self.h = h
		self.dropout_rate = dropout_rate

		# Definicion de las capas
		self.multi_head_attention_layer = MultiHeadAttention(
			d_model=self.d_model, h=self.h, dropout_rate=self.dropout_rate
		)
		self.residual_layer_1 = ResidualConnection(
			features=d_model, dropout_rate=self.dropout_rate
		)
		self.feed_forward_layer = FeedForward(
			d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
		)
		self.residual_layer_2 = ResidualConnection(
			features=d_model, dropout_rate=self.dropout_rate
		)

	def forward(
		self, input_tensor: torch.Tensor, mask: torch.Tensor | None = None
	) -> torch.Tensor:
		"""_summary_

		Args:
			input_tensor (torch.Tensor): _description_
			mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# Utilizamos self-attention, por lo que k, q, v son del mismo vector de entrada
		input_tensor = self.residual_layer_1(
			input_tensor,
			lambda x: self.multi_head_attention_layer(k=x, q=x, v=x, mask=mask),
		)

		# Segunda conexión residual con feed-forward
		input_tensor = self.residual_layer_2(
			input_tensor, lambda x: self.feed_forward_layer(x)
		)

		return input_tensor


class DecoderBlock(nn.Module):
	def __init__(self, d_model: int, d_ff: int, h: int, dropout_rate: float) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			d_ff (int): _description_
			h (int): _description_
			dropout_rate (float): _description_
		"""

		super().__init__()

		# Parametros
		self.d_model = d_model
		self.d_ff = d_ff
		self.h = h
		self.dropout_rate = dropout_rate

		self.masked_multi_head_attention_layer = MultiHeadAttention(
			d_model=self.d_model, h=self.h, dropout_rate=self.dropout_rate
		)
		self.residual_layer_1 = ResidualConnection(
			features=d_model, dropout_rate=self.dropout_rate
		)
		self.multi_head_attention_layer = MultiHeadAttention(
			d_model=self.d_model, h=self.h, dropout_rate=self.dropout_rate
		)
		self.residual_layer_2 = ResidualConnection(
			features=d_model, dropout_rate=self.dropout_rate
		)
		self.feed_forward_layer = FeedForward(
			d_model=self.d_model, d_ff=self.d_ff, dropout_rate=self.dropout_rate
		)
		self.residual_layer_3 = ResidualConnection(
			features=d_model, dropout_rate=self.dropout_rate
		)

	def forward(
		self,
		decoder_input: torch.Tensor,
		encoder_output: torch.Tensor,
		src_mask: torch.Tensor | None = None,  # Máscara para el encoder (padding)
		tgt_mask: torch.Tensor | None = None,  # Máscara causal para el decoder
	) -> torch.Tensor:
		"""_summary_

		Args:
			decoder_input (torch.Tensor): _description_
			encoder_output (torch.Tensor): _description_
			src_mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# Utilizamos self-attention, por lo que k, q, v son del mismo vector de entrada
		decoder_input = self.residual_layer_1(
			decoder_input,
			lambda x: self.masked_multi_head_attention_layer(
				k=x, q=x, v=x, mask=tgt_mask
			),
		)

		# Aquí tenemos que hacer cross-attention, usamos como K, V los encoder
		# y Q del decoder
		decoder_input = self.residual_layer_2(
			decoder_input,
			lambda x: self.multi_head_attention_layer(
				k=encoder_output, q=x, v=encoder_output, mask=src_mask
			),
		)

		decoder_output = self.residual_layer_3(
			decoder_input, lambda x: self.feed_forward_layer(x)
		)

		return decoder_output


class ProjectionLayer(nn.Module):
	# Esto permite convertir de d_model al vocab_size de nuevo

	def __init__(self, d_model: int, vocab_size: int) -> None:
		"""_summary_

		Args:
			d_model (int): _description_
			vocab_size (int): _description_
		"""

		super().__init__()

		self.d_model = d_model
		self.vocab_size = vocab_size

		self.projection_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

	def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
		"""_summary_

		Args:
			input_tensor (torch.Tensor): _description_

		Returns:
			torch.Tensor: _description_
		"""

		return self.projection_layer(input_tensor)


class Transformer(nn.Module):
	def __init__(
		self,
		src_vocab_size: int,
		tgt_vocab_size: int,
		src_seq_len: int,
		tgt_seq_len: int,
		num_encoders: int,
		num_decoders: int,
		d_model: int,
		d_ff: int,
		h: int,
		dropout_rate: float,
	) -> None:
		"""_summary_

		Args:
			src_vocab_size (int): _description_
			tgt_vocab_size (int): _description_
			src_seq_len (int): _description_
			tgt_seq_len (int): _description_
			num_encoders (int): _description_
			num_decoders (int): _description_
			d_model (int): _description_
			d_ff (int): _description_
			h (int): _description_
			dropout_rate (float): _description_
		"""

		super().__init__()

		# Parámetros
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.src_seq_len = src_seq_len
		self.tgt_seq_len = tgt_seq_len
		self.num_encoders = num_encoders
		self.num_decoders = num_decoders
		self.d_model = d_model
		self.d_ff = d_ff
		self.h = h
		self.dropout_rate = dropout_rate

		# Embeddings y Positional Encoding
		self.src_embedding = InputEmbedding(
			d_model=self.d_model, vocab_size=self.src_vocab_size
		)
		self.tgt_embedding = InputEmbedding(
			d_model=self.d_model, vocab_size=self.tgt_vocab_size
		)
		self.src_positional_encoding = PositionalEncoding(
			d_model=self.d_model,
			sequence_length=self.src_seq_len,
			dropout_rate=self.dropout_rate,
		)
		self.tgt_positional_encoding = PositionalEncoding(
			d_model=self.d_model,
			sequence_length=self.tgt_seq_len,
			dropout_rate=self.dropout_rate,
		)

		# Capas del Encoder
		self.encoder_layers = nn.ModuleList(
			[
				EncoderBlock(
					d_model=self.d_model,
					d_ff=self.d_ff,
					h=self.h,
					dropout_rate=self.dropout_rate,
				)
				for _ in range(self.num_encoders)
			]
		)

		# Capas del Decoder
		self.decoder_layers = nn.ModuleList(
			[
				DecoderBlock(
					d_model=self.d_model,
					d_ff=self.d_ff,
					h=self.h,
					dropout_rate=self.dropout_rate,
				)
				for _ in range(self.num_decoders)
			]
		)

		# Capa de proyección final
		self.projection_layer = ProjectionLayer(
			d_model=self.d_model, vocab_size=self.tgt_vocab_size
		)

	def encode(
		self, encoder_input: torch.Tensor, src_mask: torch.Tensor | None = None
	) -> torch.Tensor:
		"""_summary_

		Args:
			encoder_input (torch.Tensor): _description_
			src_mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# Aplicar embedding y positional encoding
		x = self.src_embedding(encoder_input)
		x = self.src_positional_encoding(x)

		# Pasar por todas las capas del encoder
		for encoder_layer in self.encoder_layers:
			x = encoder_layer(input_tensor=x, mask=src_mask)

		return x

	def decode(
		self,
		decoder_input: torch.Tensor,
		encoder_output: torch.Tensor,
		src_mask: torch.Tensor | None = None,
		tgt_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""_summary_

		Args:
			decoder_input (torch.Tensor): _description_
			encoder_output (torch.Tensor): _description_
			src_mask (torch.Tensor | None, optional): _description_. Defaults to None.
			tgt_mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# Aplicar embedding y positional encoding
		x = self.tgt_embedding(decoder_input)
		x = self.tgt_positional_encoding(x)

		# Pasar por todas las capas del decoder
		for decoder_layer in self.decoder_layers:
			x = decoder_layer(
				decoder_input=x,
				encoder_output=encoder_output,
				src_mask=src_mask,
				tgt_mask=tgt_mask,
			)

		return x

	def forward(
		self,
		src: torch.Tensor,
		tgt: torch.Tensor,
		src_mask: torch.Tensor | None = None,
		tgt_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""_summary_

		Args:
			src (torch.Tensor): _description_
			tgt (torch.Tensor): _description_
			src_mask (torch.Tensor | None, optional): _description_. Defaults to None.
			tgt_mask (torch.Tensor | None, optional): _description_. Defaults to None.

		Returns:
			torch.Tensor: _description_
		"""

		# Encoder
		encoder_output = self.encode(src, src_mask)

		# Decoder
		decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

		# Projection
		return self.projection_layer(decoder_output)
