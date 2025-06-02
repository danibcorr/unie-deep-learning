# Standard libraries
import math

# 3pps
import torch
from torch import nn
from torch.nn import functional as F


class Patches(nn.Module):
    def __init__(
        self,
        patch_size_height: int,
        patch_size_width: int,
        img_height: int,
        img_width: int,
    ) -> None:
        super().__init__()

        if img_height % patch_size_height != 0:
            raise ValueError(
                "img_height tiene que se divisible entre el patch_size_height"
            )

        if img_width % patch_size_width != 0:
            raise ValueError(
                "img_width tiene que se divisible entre el patch_size_width"
            )

        self.patch_size_height = patch_size_height
        self.patch_size_width = patch_size_width
        self.unfold = nn.Unfold(
            kernel_size=(self.patch_size_height, self.patch_size_width),
            stride=(self.patch_size_height, self.patch_size_width),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # unfold devuelve (b, c * patch_height * patch_width, num_patches)
        patches = self.unfold(input_tensor)
        # Necesitamos (B, NUM_PATCHES, C * patch_size_height * patch_size_width)
        return patches.transpose(2, 1)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size_height: int,
        patch_size_width: int,
        in_channels: int,
        d_model: int,
    ) -> None:
        """_summary_

        Args:
            d_model (int): _description_
            vocab_size (int): _description_
        """

        # Constructor de la clase
        super().__init__()

        # Definimos los parámetros de la clase
        self.patch_size_height = patch_size_height
        self.patch_size_width = patch_size_width
        self.in_channels = in_channels
        self.d_model = d_model

        self.embedding = nn.Linear(
            in_features=self.in_channels
            * self.patch_size_height
            * self.patch_size_width,
            out_features=self.d_model,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input_tensor (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        return self.embedding(input_tensor)


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
            nn.GELU(),
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
        if d_model % h != 0:
            raise ValueError("d_model ha de ser divisible entre h")

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


class VIT(nn.Module):
    def __init__(
        self,
        patch_size_height: int,
        patch_size_width: int,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_encoders: int,
        d_model: int,
        d_ff: int,
        h: int,
        num_classes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.patch_size_height = patch_size_height
        self.patch_size_width = patch_size_width
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_encoders = num_encoders
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Número de patches
        self.num_patches = (img_height // patch_size_height) * (
            img_width // patch_size_width
        )

        # AÑADIDO: CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        self.patch_layer = Patches(
            patch_size_height=self.patch_size_height,
            patch_size_width=self.patch_size_width,
            img_height=self.img_height,
            img_width=self.img_width,
        )

        self.embeddings = PatchEmbedding(
            patch_size_height=self.patch_size_height,
            patch_size_width=self.patch_size_width,
            in_channels=self.in_channels,
            d_model=self.d_model,
        )

        # Entiendo que la longitud de la secuencia coincide con el numero de patches
        # y un embedding más de la clase?
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            sequence_length=self.num_patches + 1,
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

        self.layer_norm = LayerNormalization(features=self.d_model)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=self.d_model, out_features=num_classes),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Extraemos los patches
        input_patches = self.patch_layer(input_tensor)

        # Convertimso a embeddings los patches
        patch_embeddings = self.embeddings(input_patches)

        # Tenemos que añadir el token de la clase al inicio de la secuencia
        # (B, 1, d_model)
        cls_tokens = self.cls_token.expand(input_tensor.shape[0], -1, -1)
        # (B, num_patches+1, d_model)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Añadir positional encoding
        embeddings = self.positional_encoding(embeddings)

        # Encoders del transformer
        encoder_output = embeddings
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)

        # Usar solo el CLS token para clasificación
        encoder_output = self.layer_norm(encoder_output)
        cls_output = encoder_output[:, 0]

        # Clasificación final
        return self.mlp_classifier(cls_output)


if __name__ == "__main__":
    model = VIT(
        patch_size_height=16,
        patch_size_width=16,
        img_height=224,
        img_width=224,
        in_channels=3,
        num_encoders=12,
        d_model=768,
        d_ff=3072,
        h=12,
        num_classes=1000,
        dropout_rate=0.1,
    )

    # Tensor de ejemplo
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Debería ser (2, 1000)
