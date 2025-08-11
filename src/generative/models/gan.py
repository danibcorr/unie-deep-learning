# 3pps
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def show_generated_samples(
    generator: nn.Module, noise, device: str, num_samples: int = 16
) -> None:
    """Función auxiliar para mostrar muestras generadas"""
    generator.eval()
    with torch.no_grad():
        samples = generator(noise[:num_samples]).cpu()
        samples = (samples + 1) / 2  # Desnormalizar de [-1,1] a [0,1]

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(num_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(samples[i, 0], cmap="gray")
            axes[row, col].axis("off")
        plt.tight_layout()
        plt.show()


class Discriminator(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size: int = 64, dropout_rate: float = 0.2
    ) -> None:
        # Constructor de la clase
        super().__init__()

        # Definimos los parámetros de la clase
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Creamos el modelo que lo ajustaremos para MNIST
        self.model = nn.Sequential(
            # Input MNIST = (B, C, H, W) = (B, 1, 28, 28) -> (B, H, 14, 14)
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout2d(p=self.dropout_rate),
            # (B, H, 14, 14) -> (B, H, 7, 7)
            nn.Conv2d(
                in_channels=self.hidden_size // 2,
                out_channels=self.hidden_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.hidden_size),
            nn.GELU(),
            nn.Dropout2d(p=self.dropout_rate),
            # (B, H, 7, 7) -> (B, 1, 7, 7)
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            # (B, 1, 7, 7) -> (B, 1, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            # (B, 1, 1, 1) -> (B, 1),
            nn.Flatten(),
            nn.Dropout(p=self.dropout_rate),
            # El discriminador ha de devolver un escalar entre 0 y 1 (falso/real)
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


class Generator(nn.Module):
    def __init__(
        self, z_dim: int, data_shape: tuple[int, int, int], hidden_size: int
    ) -> None:
        # Constructor de la clase
        super().__init__()

        # Definimos los parámetros de la clase
        self.z_dim = z_dim
        self.data_shape = data_shape
        self.hidden_size = hidden_size

        # Del ruido que es un tensor plano, vamos a crear una matriz inicial
        # de (B, H, 7, 7) que es el tamaño antes de aplanar en el Discriminador
        self.projection = nn.Sequential(
            # (B, z_dim) -> (B, H * 7 * 7)
            nn.Linear(
                in_features=self.z_dim,
                out_features=self.hidden_size * 7 * 7,
                bias=False,
            ),
        )

        self.model = nn.Sequential(
            # (B, H, 7, 7) -> (B, H, 14, 14)
            nn.ConvTranspose2d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.hidden_size),
            nn.GELU(),
            # (B, H, 14, 14) -> (B, H, 28, 28)
            nn.ConvTranspose2d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=self.hidden_size),
            nn.GELU(),
            # (B, H, 28, 28) -> (B, 1, 28, 28)
            nn.Conv2d(
                in_channels=self.hidden_size, out_channels=1, kernel_size=1, stride=1
            ),
            # Normalizamos los valores entre -1 y 1
            nn.Tanh(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # (B, z_dim) -> (B, H * 7 * 7)
        projection = self.projection(input_tensor)
        # (B, H * 7 * 7) -> (B, H, 7, 7)
        projection = projection.view(projection.size(0), self.hidden_size, 7, 7)

        return self.model(projection)


class GAN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size_discriminator: int,
        dropout_rate: float,
        z_dim: int,
        data_shape: tuple[int, int, int],
        hidden_size_generator: int,
    ) -> None:
        # Constructor de la clase
        super().__init__()

        # Definimos los parámetros
        self.in_channels = in_channels
        self.hidden_size_discriminator = hidden_size_discriminator
        self.dropout_rate = dropout_rate
        self.z_dim = z_dim
        self.data_shape = data_shape
        self.hidden_size_generator = hidden_size_generator

        # Definimos los modelos
        self.discriminator = Discriminator(
            in_channels=self.in_channels,
            hidden_size=self.hidden_size_discriminator,
            dropout_rate=self.dropout_rate,
        )
        self.generator = Generator(
            z_dim=self.z_dim,
            data_shape=self.data_shape,
            hidden_size=self.hidden_size_generator,
        )

    def forward(self, real_data: torch.Tensor, batch_size: int | None = None) -> dict:
        if batch_size is None:
            batch_size = real_data.size(0)

        # Generar ruido aleatorio
        noise = torch.randn(batch_size, self.z_dim, device=real_data.device)

        # Generar imágenes falsas
        fake_data = self.generator(noise)

        # Pasar datos reales por el discriminador
        real_predictions = self.discriminator(real_data)

        # Pasar datos falsos por el discriminador
        fake_predictions = self.discriminator(fake_data)

        return {
            "real_predictions": real_predictions,
            "fake_predictions": fake_predictions,
            "fake_data": fake_data,
            "noise": noise,
        }

    def generate_samples_inference(self, num_samples: int, device: str) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # (Num_samples, z_dim)
            noise = torch.randn(num_samples, self.z_dim, device=device)
            # (Num_samples, z_dim) -> MNIST: (Num_samples, 1, 28, 28)
            samples = self.generator(input_tensor=noise)

        return samples

    def discriminator_loss(
        self, real_predictions: torch.Tensor, fake_predictions: torch.Tensor
    ) -> torch.Tensor:
        criterion = nn.BCELoss()
        # Matriz de 1s
        real_labels = torch.ones_like(real_predictions)
        # Matriz de 0s
        fake_labels = torch.zeros_like(fake_predictions)

        real_loss = criterion(real_predictions, real_labels)
        fake_loss = criterion(fake_predictions, fake_labels)

        return (real_loss + fake_loss) / 2

    def generator_loss(self, fake_predictions: torch.Tensor) -> torch.Tensor:
        criterion = nn.BCELoss()
        fake_real_labels = torch.ones_like(fake_predictions)
        return criterion(fake_predictions, fake_real_labels)


if __name__ == "__main__":
    # Seleccionamos el dispositivo actual
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Las GANs son muy sensibles a los hiperparámetros
    lr = 2e-4
    # Dimensión de los datos de entrada (C, H, W)
    data_dimension = (1, 28, 28)
    # Esta es la dimension del ruido
    z_dim = 100
    batch_size = 128
    num_epochs = 50

    # Definimos el modelo
    model = GAN(
        in_channels=data_dimension[0],
        hidden_size_discriminator=64,
        dropout_rate=0.2,
        z_dim=z_dim,
        data_shape=data_dimension,
        hidden_size_generator=256,
    ).to(device)

    # Transformaciones que vamos a aplicar a MNIST
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                # Normalizar a [-1, 1] para coincidir con Tanh
                (0.5,),
                (0.5,),
            ),
        ]
    )

    # Descargamos MNIST
    dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Vamos a utilizar un optimizador para cada modelo
    opt_disc = optim.AdamW(params=model.discriminator.parameters(), lr=lr)
    opt_gen = optim.AdamW(params=model.generator.parameters(), lr=lr)

    # Ruido fijo para generar muestras durante el entrenamiento
    fixed_noise = torch.randn(64, z_dim, device=device)

    # Listas para guardar las pérdidas
    disc_losses = []
    gen_losses = []

    print("Iniciando entrenamiento de la GAN...")

    for epoch in range(num_epochs):
        epoch_disc_loss: int | float = 0
        epoch_gen_loss: int | float = 0

        pbar = tqdm(loader, desc=f"Época {epoch + 1}/{num_epochs}")

        for _, (real, _) in enumerate(pbar):
            real = real.to(device)
            batch_size_current = real.shape[0]

            # ENTRENAR DISCRIMINADOR
            opt_disc.zero_grad()

            # Generar datos falsos
            noise = torch.randn(batch_size_current, z_dim, device=device)
            fake_data = model.generator(
                noise
            ).detach()  # Detach para no actualizar generador

            # Predicciones del discriminador
            real_preds = model.discriminator(real)
            fake_preds = model.discriminator(fake_data)

            # Pérdida del discriminador
            lossD = model.discriminator_loss(real_preds, fake_preds)
            lossD.backward()
            opt_disc.step()

            # ENTRENAR GENERADOR
            opt_gen.zero_grad()

            # Generar nuevos datos falsos (sin detach)
            noise = torch.randn(batch_size_current, z_dim, device=device)
            fake_data = model.generator(noise)
            fake_preds_for_gen = model.discriminator(fake_data)

            # Pérdida del generador
            lossG = model.generator_loss(fake_preds_for_gen)
            lossG.backward()
            opt_gen.step()

            # Acumular pérdidas
            epoch_disc_loss += lossD.item()
            epoch_gen_loss += lossG.item()

            # Actualizar barra de progreso
            pbar.set_postfix(
                {"D_loss": f"{lossD.item():.4f}", "G_loss": f"{lossG.item():.4f}"}
            )

        # Pérdidas promedio de la época
        avg_disc_loss = epoch_disc_loss / len(loader)
        avg_gen_loss = epoch_gen_loss / len(loader)
        disc_losses.append(avg_disc_loss)
        gen_losses.append(avg_gen_loss)

        print(
            f"Época {epoch + 1}, "
            f"Pérdida D: {avg_disc_loss:.4f}, "
            f"Pérdida G: {avg_gen_loss:.4f}"
        )

        # Mostrar muestras cada 5 épocas
        if (epoch + 1) % 5 == 0:
            print(f"\nMostrando muestras generadas en época {epoch + 1}...")
            show_generated_samples(model.generator, fixed_noise, device, num_samples=16)

    print("\n¡Entrenamiento completado!")

    # Mostrar gráfica de pérdidas
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(disc_losses, label="Discriminador", color="red")
    plt.plot(gen_losses, label="Generador", color="blue")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Evolución de las Pérdidas")
    plt.legend()
    plt.grid(True)

    # Mostrar muestras generadas finales
    plt.subplot(1, 2, 2)
    model.generator.eval()
    with torch.no_grad():
        final_samples = model.generator(fixed_noise[:16]).cpu()
        final_samples = (final_samples + 1) / 2  # Desnormalizar
        grid = torchvision.utils.make_grid(final_samples, nrow=4, padding=2)
        plt.imshow(grid.permute(1, 2, 0), cmap="gray")
        plt.title("Muestras Finales Generadas")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nMostrando muestras generadas...")
    show_generated_samples(model.generator, fixed_noise, device, num_samples=16)
