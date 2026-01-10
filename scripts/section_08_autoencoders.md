# Autoencoders

Los autoencoders constituyen una familia de arquitecturas de redes neuronales diseñadas
para aprender representaciones comprimidas de los datos de forma no supervisada. La
estructura fundamental de un autoencoder se organiza en dos bloques principales: Un
codificador (encoder), que transforma la entrada original en una representación latente
de menor dimensión, y un decodificador (decoder), que toma dicha representación latente y
reconstruye a partir de ella una aproximación de la entrada original. El objetivo de
entrenamiento consiste en minimizar la discrepancia entre la salida reconstruida y la
entrada, de modo que el modelo se vea obligado a capturar las características más
relevantes de los datos en el espacio latente.

En este documento se presentan diversas variantes de autoencoders, desde arquitecturas
densas básicas hasta modelos más avanzados como los variational autoencoders (VAE), los
Beta-VAE y los VQ-VAE. Todas las implementaciones se realizan sobre el conjunto de datos
MNIST y se proporcionan en forma de código funcional, de principio a fin, listo para
ejecutarse en un entorno tipo Jupyter Notebook.

## Autoencoder _vanilla_ con capas densas

El autoencoder _vanilla_ utiliza exclusivamente capas densas (_fully connected_) para
codificar y decodificar las imágenes de MNIST. Cada imagen de tamaño $28 \times 28$ se
aplana a un vector de dimensión 784 y se proyecta a un espacio latente de menor
dimensión. El codificador aplica una secuencia de transformaciones lineales y funciones
de activación no lineales hasta alcanzar el espacio latente, mientras que el
decodificador realiza el proceso inverso para reconstruir la imagen.

Esta configuración introduce la idea central de los autoencoders, pero presenta
limitaciones claras. Las capas densas no aprovechan explícitamente la estructura espacial
de la imagen, lo que conduce a un número elevado de parámetros debido a la conectividad
completa entre neuronas. Además, al no modelar de forma específica las relaciones locales
entre píxeles, las reconstrucciones tienden a ser más borrosas y menos detalladas.

A continuación se presenta una implementación funcional básica sobre MNIST.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 32) -> None:
        super().__init__()

        # Codificador: Reduce progresivamente las dimensiones
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: Reconstruye desde el espacio latente
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Salida en [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplanar la imagen
        x = x.view(x.size(0), -1)
        # Codificar
        latent = self.encoder(x)
        # Decodificar
        reconstructed = self.decoder(latent)
        # Volver a forma de imagen
        return reconstructed.view(-1, 1, 28, 28)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.encoder(x)
```

```python
def prepare_mnist_data(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
```

```python
def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    device: str = "cuda"
) -> nn.Module:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return model
```

```python
def visualize_reconstructions(
    model: nn.Module,
    test_loader: DataLoader,
    num_images: int = 10,
    device: str = "cuda"
) -> None:
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)

    with torch.no_grad():
        reconstructed = model(data)

    data = data.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstruida")

    plt.tight_layout()
    plt.show()
```

```python
# Ejecución del autoencoder vanilla
train_loader, test_loader = prepare_mnist_data()
vanilla_ae = VanillaAutoencoder(input_dim=784, latent_dim=32)
device = "cuda" if torch.cuda.is_available() else "cpu"

vanilla_ae = train_autoencoder(vanilla_ae, train_loader, num_epochs=10, device=device)
visualize_reconstructions(vanilla_ae, test_loader, device=device)
```

## Denoising Autoencoder

El _denoising autoencoder_ extiende el planteamiento anterior al introducir ruido en la
entrada durante el entrenamiento. En este caso, el codificador recibe una versión
corrupta de la imagen, mientras que la función de pérdida compara la salida del
decodificador con la imagen original limpia. Este mecanismo fuerza al modelo a aprender
representaciones latentes robustas que capturan la estructura subyacente de los datos, en
lugar de limitarse a aproximar la función identidad.

El ruido se introduce habitualmente como ruido gaussiano aditivo, recortando
posteriormente los valores para mantenerlos en el rango $[0, 1]$. De este modo, el modelo
aprende a “deshacer” la corrupción, actuando como un filtro que conserva contenido
relevante y descarta detalles espurios.

A continuación se muestra una implementación de esta variante sobre MNIST.

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed.view(-1, 1, 28, 28)
```

```python
def add_noise(images: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    noisy = images + noise_factor * torch.randn_like(images)
    noisy = torch.clip(noisy, 0.0, 1.0)
    return noisy
```

```python
def train_denoising_ae(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    device: str = "cuda",
    noise_factor: float = 0.3
) -> nn.Module:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in train_loader:
            clean_data = data.to(device)
            noisy_data = add_noise(clean_data, noise_factor)

            optimizer.zero_grad()
            reconstructed = model(noisy_data)
            loss = criterion(reconstructed, clean_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return model
```

```python
def visualize_denoising(
    model: nn.Module,
    test_loader: DataLoader,
    noise_factor: float = 0.3,
    num_images: int = 10,
    device: str = "cuda"
) -> None:
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)
    noisy_data = add_noise(data, noise_factor)

    with torch.no_grad():
        reconstructed = model(noisy_data)

    data = data.cpu()
    noisy_data = noisy_data.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", rotation=0, labelpad=40)

        axes[1, i].imshow(noisy_data[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Noisy", rotation=0, labelpad=40)

        axes[2, i].imshow(reconstructed[i].squeeze(), cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Denoised", rotation=0, labelpad=40)

    plt.tight_layout()
    plt.show()
```

```python
# Ejecución del denoising autoencoder
denoising_ae = DenoisingAutoencoder(input_dim=784, latent_dim=32)
denoising_ae = train_denoising_ae(denoising_ae, train_loader, num_epochs=10, device=device)
visualize_denoising(denoising_ae, test_loader, device=device)
```

## Autoencoder convolucional

Los autoencoders convolucionales se adaptan mejor a datos de tipo imagen porque explotan
de forma explícita la estructura espacial. El codificador aplica convoluciones con pesos
compartidos y filtros locales; la dimensionalidad espacial se reduce mediante _stride_ y
la acumulación de capas. El decodificador utiliza convoluciones transpuestas para
realizar operaciones de _upsampling_ y reconstruir la resolución original.

En este contexto, las convoluciones proporcionan varias ventajas. Reducen
significativamente el número de parámetros respecto a las capas densas, al compartir
pesos entre distintas posiciones espaciales. Además, capturan patrones locales y
estructuras jerárquicas (bordes, partes de dígitos, dígitos completos), lo que conduce a
reconstrucciones más nítidas y coherentes con el contenido de las imágenes.

La siguiente implementación ilustra un autoencoder convolucional con un “cuello de
botella” lineal intermedio.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()

        # Codificador convolucional
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 7x7 -> 4x4 (ligera reducción adicional)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # Bottleneck lineal
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))

        # Decoder con convoluciones transpuestas
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3, stride=2,
                padding=1, output_padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 7x7 -> 14x14
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc_encode(x)

        # Decode
        x = self.fc_decode(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)

        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        return self.fc_encode(x)
```

```python
# Entrenamiento del autoencoder convolucional
conv_ae = ConvAutoencoder(latent_dim=128)
conv_ae = train_autoencoder(conv_ae, train_loader, num_epochs=10, device=device)
visualize_reconstructions(conv_ae, test_loader, device=device)
```

Las convoluciones transpuestas pueden introducir artefactos característicos denominados
_checkerboard artifacts_ o patrones de tablero de ajedrez, que surgen cuando la
combinación de tamaño de _kernel_ y _stride_ produce superposiciones desiguales en la
operación de _upsampling_.

## Autoencoder con _upsampling_ basado en interpolación

Para mitigar los artefactos de tipo _checkerboard_ es habitual sustituir las
convoluciones transpuestas por una estrategia de _upsampling_ basada en interpolación
seguida de convoluciones estándar. En este esquema, se aumenta primero la resolución
espacial mediante interpolación (bilineal, bicúbica, etc.) y, posteriormente, se aplica
una convolución para refinar el resultado y aprender filtros sobre la imagen reescalada.

Este procedimiento tiende a generar reconstrucciones más suaves y visualmente coherentes,
reduciendo notablemente los patrones indeseados, a costa de un cierto incremento en el
coste computacional.

El modelo siguiente mantiene el mismo codificador convolucional del autoencoder anterior,
pero reemplaza el decodificador basado en `ConvTranspose2d` por un decodificador que
combina `Upsample` y `Conv2d`.

```python
class UpsamplingAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()

        # Codificador idéntico al convolucional
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))

        # Decoder con upsampling + convolución
        self.decoder = nn.Sequential(
            # 4x4 -> 7x7
            nn.Upsample(size=(7, 7), mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # 7x7 -> 14x14
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.fc_encode(x)
        x = self.fc_decode(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)
        return reconstructed
```

```python
# Entrenamiento y visualización
upsampling_ae = UpsamplingAutoencoder(latent_dim=128)
upsampling_ae = train_autoencoder(upsampling_ae, train_loader, num_epochs=10, device=device)
visualize_reconstructions(upsampling_ae, test_loader, device=device)
```

La utilización de interpolación bilineal o bicúbica seguida de convoluciones estándar
produce, en general, reconstrucciones más agradables visualmente y reduce de forma
significativa los artefactos de tipo _checkerboard_, manteniendo la capacidad del modelo
para capturar patrones de alto nivel.

## Variational Autoencoder (VAE)

El variational autoencoder (VAE) introduce un cambio conceptual importante respecto a los
autoencoders deterministas. En lugar de aprender un mapeo directo de la entrada a un
vector latente fijo, el codificador aprende los parámetros de una distribución de
probabilidad sobre el espacio latente. Habitualmente se asume que cada dimensión latente
sigue una distribución gaussiana independiente, de modo que el codificador produce una
media $\mu$ y un logaritmo de la varianza $\log \sigma^2$ para cada dimensión.

Durante el entrenamiento, se genera una muestra $z$ del espacio latente utilizando el
denominado _reparameterization trick_: \[ z = \mu + \sigma \odot \varepsilon, \] donde
$\varepsilon \sim \mathcal{N}(0, I)$ y
$\sigma = \exp\left(\tfrac{1}{2} \log
\sigma^2\right)$. Esta formulación permite propagar
gradientes a través de la operación de muestreo.

La función de pérdida del VAE incluye dos términos. El primero es la pérdida de
reconstrucción, que mide la discrepancia entre la imagen original y la reconstruida (por
ejemplo, mediante entropía cruzada binaria). El segundo es un término de regularización
basado en la divergencia de Kullback–Leibler (KL) entre la distribución latente aprendida
y una distribución normal estándar $\mathcal{N}(0, I)$: \[ \mathcal{L}_{\text{KL}} =
-\frac{1}{2} \sum_{i} \left(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\right). \] Este
término obliga al espacio latente a adoptar una estructura bien comportada, facilitando
el muestreo y la generación de nuevas muestras.

A continuación se presenta una implementación de un VAE convolucional para MNIST.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()

        # Codificador convolucional
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
```

```python
def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    recon_loss = nn.functional.binary_cross_entropy(
        reconstructed, original, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```

```python
def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    device: str = "cuda"
) -> nn.Module:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {total_loss / len(train_loader.dataset):.4f}"
        )

    return model
```

```python
def visualize_latent_space_tsne(
    model: VAE,
    data_loader: DataLoader,
    device: str = "cuda",
    n_samples: int = 5000
) -> None:
    """Visualiza el espacio latente usando t-SNE."""
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())

            if len(latent_vectors) * data.size(0) >= n_samples:
                break

    latent_vectors = np.concatenate(latent_vectors, axis=0)[:n_samples]
    labels = np.concatenate(labels, axis=0)[:n_samples]

    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
        s=5
    )
    plt.colorbar(scatter, label="Dígito")
    plt.title("Visualización t-SNE del espacio latente del VAE")
    plt.xlabel("t-SNE dimensión 1")
    plt.ylabel("t-SNE dimensión 2")
    plt.tight_layout()
    plt.show()
```

```python
def generate_samples(
    model: VAE,
    num_samples: int = 16,
    latent_dim: int = 20,
    device: str = "cuda"
) -> None:
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z).cpu()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
```

```python
def prepare_mnist_data(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
```

```python
# Preparar datos
train_loader, test_loader = prepare_mnist_data()

# Entrenar VAE
vae = VAE(latent_dim=20)
device = "cuda" if torch.cuda.is_available() else "cpu"
vae = train_vae(vae, train_loader, num_epochs=20, device=device)

# Visualizar espacio latente con t-SNE
visualize_latent_space_tsne(vae, test_loader, device=device)

# Generar muestras sintéticas
generate_samples(vae, latent_dim=20, device=device)

# Visualizar reconstrucciones
with torch.no_grad():
    data, _ = next(iter(test_loader))
    data = data[:10].to(device)
    reconstructed, _, _ = vae(data)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", size=12)
    axes[1, 0].set_ylabel("Reconstruido", size=12)
    plt.tight_layout()
    plt.show()
```

Los VAE resultan especialmente útiles para la generación de datos sintéticos, mediante
muestreo directo en el espacio latente, y para detección de anomalías, analizando
ejemplos fuera de distribución. No obstante, pueden sufrir el fenómeno de _posterior
collapse_, en el que el decodificador ignora en gran medida la información latente y
aprende a reconstruir a partir de patrones locales, reduciendo la calidad e
informatividad de las representaciones en el espacio latente.

## Beta-VAE

El Beta-VAE introduce un hiperparámetro $\beta$ en la función de pérdida del VAE para
ponderar el término de divergencia KL: \[ \mathcal{L}_{\beta\text{-VAE}} =
\mathcal{L}_{\text{recon}} + \beta \,\mathcal{L}\_{\text{KL}}. \] Cuando $\beta > 1$, se
fuerza un mayor alineamiento de la distribución latente con la normal estándar, lo que
tiende a producir representaciones más _disentangled_. En un espacio latente
_disentangled_, cada dimensión captura preferentemente un factor de variación
independiente de los datos (por ejemplo, grosor del trazo, inclinación, tamaño), lo que
mejora la interpretabilidad y el control sobre las muestras generadas.

Valores excesivamente altos de $\beta$ pueden degradar la calidad de reconstrucción, al
penalizar en exceso la complejidad del código latente.

A continuación se muestra cómo adaptar la pérdida y el procedimiento de entrenamiento
para un Beta-VAE utilizando la arquitectura de VAE definida anteriormente.

```python
def beta_vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 4.0
) -> torch.Tensor:
    recon_loss = nn.functional.binary_cross_entropy(
        reconstructed, original, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

```python
def train_beta_vae(
    model: VAE,
    train_loader: DataLoader,
    num_epochs: int = 10,
    beta: float = 4.0,
    device: str = "cuda"
) -> VAE:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data)
            loss = beta_vae_loss(reconstructed, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model
```

Para explorar el efecto de variaciones controladas en dimensiones individuales del
espacio latente, se emplea la técnica de _latent traversal_, que consiste en modificar
sistemáticamente una única coordenada latente manteniendo fijas las restantes.

```python
def visualize_latent_traversal(
    model: VAE,
    test_loader: DataLoader,
    latent_dim: int = 20,
    dim_to_vary: int = 0,
    device: str = "cuda"
) -> None:
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[0:1].to(device)

    with torch.no_grad():
        mu, _ = model.encode(data)

        values = torch.linspace(-3, 3, 10)
        samples = []

        for val in values:
            z = mu.clone()
            z[0, dim_to_vary] = val
            reconstructed = model.decode(z)
            samples.append(reconstructed)

        samples = torch.cat(samples, dim=0)

    samples = samples.cpu()

    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{values[i]:.1f}")

    plt.tight_layout()
    plt.show()
```

```python
# Entrenamiento del Beta-VAE
beta_vae = VAE(latent_dim=20)
beta_vae = train_beta_vae(beta_vae, train_loader, num_epochs=20, beta=4.0, device=device)

# Visualizar la variación de algunas dimensiones latentes
for dim in range(5):
    visualize_latent_traversal(beta_vae, test_loader, dim_to_vary=dim, device=device)
```

La técnica de _latent traversal_ permite inspeccionar la influencia de cada dimensión
latente en las muestras generadas, facilitando la interpretación de representaciones
_disentangled_ y el diseño de manipulaciones controladas sobre atributos concretos.

## VQ-VAE (Vector Quantized VAE)

El VQ-VAE introduce una modificación fundamental en el tratamiento del espacio latente:
En lugar de utilizar códigos continuos, emplea una representación discreta basada en un
_codebook_ de _embeddings_ aprendidos. El codificador proyecta la entrada a un tensor
latente continuo de dimensión $C$; posteriormente, cada vector latente se cuantiza
seleccionando el _embedding_ más cercano del _codebook_, es decir, se asigna a un índice
discreto. El decodificador recibe los _embeddings_ cuantizados y reconstruye la entrada.

Esta discretización ofrece varias ventajas. Por un lado, evita el problema de _posterior
collapse_ habitual en algunos VAE, ya que la cuantización fuerza al modelo a utilizar
activamente el espacio latente. Por otro, la representación discreta es especialmente
adecuada para ser modelada posteriormente mediante modelos autoregresivos (por ejemplo,
transformadores), lo que ha sido clave en arquitecturas generativas como DALL·E. En este
contexto, los índices latentes actúan como _tokens_ sobre los que resultan aplicables
técnicas de modelado de lenguaje.

A continuación se presenta una implementación sencilla de VQ-VAE para MNIST, incluyendo
el módulo de cuantización vectorial.

```python
"""
VQ-VAE (Vector Quantized Variational Autoencoder) Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

```python
class VectorQuantizer(nn.Module):
    """
    Capa de Vector Quantizer para VQ-VAE.
    Convierte vectores latentes continuos en códigos discretos del codebook.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Codebook de embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / num_embeddings,
            1 / num_embeddings
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor de forma (B, C, H, W)

        Returns:
            quantized: Tensor cuantizado (B, C, H, W)
            loss: Pérdida de cuantización (codebook + compromiso)
            encoding_indices: Índices de los vectores del codebook seleccionados
        """
        # Reordenar a (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Aplanar a (B*H*W, C)
        flat_input = inputs.view(-1, self.embedding_dim)

        # Distancias L2 a cada embedding del codebook
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )

        # Índice del embedding más cercano
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Codificación one-hot
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Cuantización mediante el codebook
        quantized = torch.matmul(encodings, self.embeddings.weight).view(
            input_shape
        )

        # Pérdidas VQ
        e_latent_loss = nn.functional.mse_loss(
            quantized.detach(), inputs
        )
        q_latent_loss = nn.functional.mse_loss(
            quantized, inputs.detach()
        )
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Volver a (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, encoding_indices
```

```python
class VQVAE(nn.Module):
    """
    Modelo VQ-VAE con codificador, vector quantizer y decoder.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64
    ) -> None:
        super().__init__()

        # Codificador: (1, 28, 28) -> (embedding_dim, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=1)  # 7x7, C=embedding_dim
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        # Decoder: (embedding_dim, 7, 7) -> (1, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim, 64,
                kernel_size=4, stride=2, padding=1
            ),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1
            ),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor de entrada (B, 1, 28, 28)

        Returns:
            reconstructed: Reconstrucción (B, 1, 28, 28)
            vq_loss: Pérdida de cuantización vectorial
        """
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss

    def encode(self, x: torch.Tensor):
        """Codifica y cuantiza la entrada."""
        z = self.encoder(x)
        quantized, _, indices = self.vq(z)
        return quantized, indices

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica un tensor latente cuantizado."""
        return self.decoder(z)
```

```python
def train_vqvae(
    model: VQVAE,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda"
) -> VQVAE:
    """
    Entrena el modelo VQ-VAE.

    Args:
        model: Modelo VQVAE.
        train_loader: DataLoader de entrenamiento.
        num_epochs: Número de épocas.
        lr: Tasa de aprendizaje.
        device: "cuda" o "cpu".

    Returns:
        Modelo entrenado.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_recon_loss = 0.0
        total_vq_loss = 0.0

        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            reconstructed, vq_loss = model(data)

            recon_loss = nn.functional.mse_loss(reconstructed, data)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

        avg_recon = total_recon_loss / len(train_loader)
        avg_vq = total_vq_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Recon Loss: {avg_recon:.6f} | "
            f"VQ Loss: {avg_vq:.6f}"
        )

    return model
```

```python
def visualize_vqvae_reconstructions(
    model: VQVAE,
    test_loader: DataLoader,
    device: str = "cuda",
    num_images: int = 8
) -> None:
    """
    Visualiza imágenes originales y reconstruidas por VQ-VAE.
    """
    model.eval()

    data, _ = next(iter(test_loader))
    data = data[:num_images].to(device)

    with torch.no_grad():
        reconstructed, _ = model(data)

    data = data.cpu()
    reconstructed = reconstructed.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(12, 3))

    for i in range(num_images):
        axes[0, i].imshow(data[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", size=12)
    axes[1, 0].set_ylabel("Reconstruida", size=12)

    plt.tight_layout()
    plt.show()
```

```python
# Ejecución principal de VQ-VAE
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
NUM_EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vqvae = VQVAE(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMBEDDING_DIM)
vqvae = train_vqvae(vqvae, train_loader, num_epochs=NUM_EPOCHS, device=DEVICE)
visualize_vqvae_reconstructions(vqvae, test_loader, device=DEVICE)
```

El VQ-VAE proporciona un espacio latente discreto especialmente adecuado para su
integración en sistemas multimodales y modelos generativos complejos, en los que la
tokenización de los datos resulta fundamental. La cuantización vectorial ofrece una base
robusta para aplicar técnicas de modelado secuencial avanzadas sobre representaciones de
imagen y facilita la conexión con arquitecturas de lenguaje que operan sobre secuencias
discretas.
