# 3pps
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 256) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.res_block(input_tensor)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_residuals: int,
        hidden_size: int = 256,
        kernel_size: int = 4,
        stride: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_residuals = num_residuals
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels=hidden_size, hidden_size=hidden_size)
                for _ in range(self.num_residuals)
            ]
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        encoder_output = self.model(input_tensor)
        for res_block in self.residual_blocks:
            encoder_output = res_block(encoder_output)
        return encoder_output


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_residuals: int,
        out_channels: int = 3,  # Channel output (RGB)
        hidden_size: int = 256,
        kernel_size: int = 4,
        stride: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_residuals = num_residuals
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=self.in_channels, hidden_size=self.hidden_size
                )
                for _ in range(self.num_residuals)
            ]
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
            ),
            nn.ConvTranspose2d(
                in_channels=self.hidden_size,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=1,
            ),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        decoder_output = input_tensor
        for res_block in self.residual_blocks:
            decoder_output = res_block(decoder_output)

        return self.model(decoder_output)


class VectorQuantizer(nn.Module):
    def __init__(
        self, size_discrete_space: int, size_embeddings: int, beta: float = 0.25
    ) -> None:
        super().__init__()

        self.size_discrete_space = size_discrete_space
        self.size_embeddings = size_embeddings
        self.beta = beta

        # Definimos el codebook como una matriz de K embeddings x D tamaño de embeddings
        # Ha de ser una matriz aprendible
        self.codebook = nn.Embedding(
            num_embeddings=self.size_discrete_space, embedding_dim=self.size_embeddings
        )
        # Initialize weights uniformly
        self.codebook.weight.data.uniform_(
            -1 / self.size_discrete_space, 1 / self.size_discrete_space
        )

    def forward(
        self, encoder_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Comentario de otras implementaciones: The channels are used as the space
        # in which to quantize.
        # Encoder output ->  (B, C, H, W) -> (0, 1, 2, 3) -> (0, 2, 3, 1) -> (0*2*3, 1)
        encoder_output = encoder_output.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = encoder_output.size()
        encoder_output_flat = encoder_output.reshape(-1, c)

        # Calculamos la distancia entre ambos vectores
        distances = (
            torch.sum(encoder_output_flat**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(encoder_output_flat, self.codebook.weight.t())
        )

        # Realizamos el encoding y extendemos una dimension (B*H*W, 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Matriz de ceros de (indices, size_discrete_space)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.size_discrete_space,
            device=encoder_output.device,
        )
        # Colocamos un 1 en los indices de los encodings con el
        # valor mínimo de distancia creando un vector one-hot
        encodings.scatter_(1, encoding_indices, 1)

        # Se cuantiza colocando un cero en los pesos no relevantes (distancias grandes)
        # del codebook y le damos formato de nuevo al tensor
        quantized = torch.matmul(encodings, self.codebook.weight).view(b, h, w, c)

        # VQ-VAE loss terms
        # L = ||sg[z_e] - e||^2 + β||z_e - sg[e]||^2
        # FIX: Corrected variable names and loss calculation
        commitment_loss = F.mse_loss(
            quantized.detach(), encoder_output
        )  # ||sg[z_e] - e||^2
        embedding_loss = F.mse_loss(
            quantized, encoder_output.detach()
        )  # ||z_e - sg[e]||^2
        vq_loss = commitment_loss + self.beta * embedding_loss

        # Straight-through estimator
        quantized = encoder_output + (quantized - encoder_output).detach()

        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return (
            vq_loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
        )


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        size_discrete_space: int,
        size_embeddings: int,
        num_residuals: int,
        hidden_size: int,
        kernel_size: int,
        stride: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.size_discrete_space = size_discrete_space
        self.size_embeddings = size_embeddings
        self.num_residuals = num_residuals
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.beta = beta

        self.encoder = Encoder(
            in_channels=self.in_channels,
            num_residuals=self.num_residuals,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.decoder = Decoder(
            in_channels=self.hidden_size,
            num_residuals=self.num_residuals,
            out_channels=self.in_channels,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        self.vector_quantizer = VectorQuantizer(
            size_discrete_space=self.size_discrete_space,
            size_embeddings=self.hidden_size,
            beta=self.beta,
        )

    def forward(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(input_tensor)
        vq_loss, quantized, perplexity, _ = self.vector_quantizer(encoder_output)
        decoder_output = self.decoder(quantized)
        return vq_loss, decoder_output, perplexity


if __name__ == "__main__":
    model = VQVAE(
        in_channels=3,
        size_discrete_space=512,
        size_embeddings=64,
        num_residuals=2,
        hidden_size=128,
        kernel_size=4,
        stride=2,
        beta=0.25,
    )

    x = torch.randn(4, 3, 64, 64)
    vq_loss, reconstruction, perplexity = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")
