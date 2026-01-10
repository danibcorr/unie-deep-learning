# Mecanismos de atención en redes convolucionales

Los mecanismos de atención en redes convolucionales permiten que el modelo se centre de
manera adaptativa en las características más relevantes de la señal, ya sea a nivel de
canales o a nivel espacial. Estos módulos aprenden a recalibrar las activaciones
intermedias asignando pesos de importancia diferenciados, lo que incrementa la capacidad
de representación del modelo sin aumentar de forma drástica el número de parámetros ni el
coste computacional.

En arquitecturas modernas, la atención se integra de forma modular en bloques
convolucionales ya existentes, como los bloques residuales de ResNet. En lo que sigue se
describen y se implementan dos de los mecanismos de atención más influyentes en redes
convolucionales: el bloque _Squeeze-and-Excitation_ (SE) y el _Convolutional Block
Attention Module_ (CBAM). Posteriormente se presenta una implementación funcional
completa en PyTorch sobre CIFAR-10 que permite comparar empíricamente el impacto de cada
mecanismo en una arquitectura tipo ResNet-18.

## Bloque Squeeze-and-Excitation (SE)

El bloque _Squeeze-and-Excitation_, propuesto en el trabajo _Squeeze-and-Excitation
Networks_ (https://arxiv.org/abs/1709.01507), introduce un mecanismo de atención a nivel
de canal. La idea central consiste en modelar explícitamente las relaciones de
dependencia entre canales de características, de modo que la red aprenda a enfatizar
aquellos que resultan más informativos para la tarea, suprimiendo al mismo tiempo canales
menos relevantes o redundantes.

El mecanismo SE se descompone en dos etapas conceptuales. En la fase de _squeeze_ se
reduce la dimensión espacial de cada mapa de características mediante _global average
pooling_. Con ello, cada canal se comprime a un único valor escalar que resume su
activación global en toda la imagen. En la fase de _excitation_ estos valores agregados
se introducen en una pequeña red totalmente conectada que aprende una función de atención
por canal. La salida de esta red es un vector de pesos en el intervalo $(0, 1)$, que se
aplica multiplicativamente a los canales originales, recalibrando su importancia
relativa.

Sea $X \in \mathbb{R}^{B \times C \times H \times W}$ un bloque de características con
$B$ ejemplos, $C$ canales y dimensiones espaciales $H \times W$. La operación de
_squeeze_ calcula, para cada canal $c$,

$$
z*c = \frac{1}{HW} \sum*{i=1}^{H} \sum\_{j=1}^{W} X_c(i, j).
$$

El vector comprimido $z \in \mathbb{R}^{C}$ se procesa mediante una red de dos capas
lineales con reducción intermedia, que produce un vector de pesos $s \in (0, 1)^{C}$ tras
una activación sigmoide. La recalibración se implementa como

$$
\tilde{X}\_c(i, j) = s_c \cdot X_c(i, j).
$$

A continuación se muestra una implementación del bloque SE y su integración en un bloque
residual básico en PyTorch. El código está orientado a su uso directo en un flujo de
trabajo reproducible y ejecutable.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()

        reduced_channels = max(in_channels // reduction_ratio, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        # Squeeze: Promedio global por canal
        squeezed = self.squeeze(x).view(batch_size, channels)

        # Excitation: Pesos por canal en (0, 1)
        excited = self.excitation(squeezed).view(batch_size, channels, 1, 1)

        # Recalibración de canales
        return x * excited


class SEResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction_ratio: int = 16
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SqueezeExcitation(out_channels, reduction_ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)

        out += identity
        out = F.relu(out)

        return out
```

Para verificar la correcta construcción del bloque SE se puede emplear un pequeño test
funcional que comprueba las dimensiones de entrada y salida, así como el número de
parámetros implicados.

```python
def test_se_block() -> None:
    x = torch.randn(2, 64, 32, 32)
    se_block = SqueezeExcitation(in_channels=64, reduction_ratio=16)
    output = se_block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"SE parameters: {sum(p.numel() for p in se_block.parameters())}")

    assert x.shape == output.shape, "Shape mismatch"
    print("SE Block test passed")


test_se_block()
```

El bloque SE introduce un número adicional de parámetros relativamente moderado,
controlado por el parámetro `reduction_ratio`. Este parámetro determina el tamaño del
cuello de botella en la red de _excitation_: Valores más elevados reducen la capacidad
del módulo, pero disminuyen su coste computacional. En la práctica, configuraciones como
`reduction_ratio = 16` ofrecen un equilibrio adecuado entre capacidad de modelado y
eficiencia.

## Convolutional Block Attention Module (CBAM)

El módulo CBAM (_Convolutional Block Attention Module_) extiende la idea de SE
incorporando de forma secuencial atención tanto en el dominio de canales como en el
dominio espacial. En primer lugar, aplica un módulo de atención de canal conceptualmente
similar al de SE, pero combinando información procedente de _global average pooling_ y
_global max pooling_. Posteriormente, aplica un módulo de atención espacial que analiza
la distribución de activaciones a través de los canales para determinar qué regiones de
la imagen son más relevantes.

El módulo de atención de canal en CBAM se construye a partir de dos vías paralelas. Una
recibe como entrada la salida de un _global average pooling_ y la otra utiliza la salida
de un _global max pooling_, ambos calculados a nivel de canal. Cada uno de estos
resúmenes se procesa con una pequeña red convolucional de tamaño $1 \times 1$, que actúa
como proyección totalmente conectada compartida. Las dos salidas resultantes se combinan
mediante suma y, a continuación, se aplica una función sigmoide para obtener un mapa de
atención de canal que modula la contribución de cada canal en la representación.

El módulo de atención espacial se aplica sobre la salida ya recalibrada por canal. Para
ello, se calculan dos mapas espaciales monocanal a partir de la agregación sobre el eje
de canales mediante media y máximo. Estos dos mapas se concatenan y se procesan con una
convolución de tamaño $k \times k$, típicamente con $k = 7$, seguida de una sigmoide. El
resultado es un mapa de atención espacial que se aplica multiplicativamente a la señal,
modulando la importancia de cada posición $(i, j)$ en la imagen.

A continuación se presenta la implementación de CBAM (atención de canal y espacial) y su
integración en un bloque residual.

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()

        reduced_channels = max(in_channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels, reduction_ratio
        )
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CBAMResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAM(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.cbam(out)

        out += identity
        out = F.relu(out)

        return out
```

El siguiente fragmento de código realiza una comprobación básica del módulo CBAM, análoga
al test aplicado en el caso del bloque SE.

```python
def test_cbam() -> None:
    x = torch.randn(2, 64, 32, 32)
    cbam = CBAM(in_channels=64)
    output = cbam(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"CBAM parameters: {sum(p.numel() for p in cbam.parameters())}")

    assert x.shape == output.shape, "Shape mismatch"
    print("CBAM test passed")


test_cbam()
```

En la práctica, CBAM suele proporcionar mejoras consistentes sobre SE, ya que combina
atención a nivel de canal y atención espacial. La atención espacial resulta especialmente
útil en tareas donde la localización de objetos o de regiones discriminativas desempeña
un papel crítico, como la detección y la segmentación de objetos, o el reconocimiento en
escenarios con múltiples instancias por imagen.

## Implementación funcional completa en CIFAR-10

En esta sección se presenta una implementación funcional en PyTorch que integra los
mecanismos de atención Squeeze-and-Excitation y CBAM en una arquitectura tipo ResNet-18,
y compara su rendimiento sobre el conjunto de datos CIFAR-10. Se parte de una ResNet-18
base sin atención, sobre la que se construyen variantes con bloques SE y con bloques
CBAM. El código se organiza como un script ejecutable, fácilmente convertible en un
cuaderno Jupyter y diseñado para ser reproducible de principio a fin.

```python
"""
Attention Mechanisms in CNNs - Versión simplificada
SE Block y CBAM con entrenamiento en CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
```

### Configuración y carga de datos

En primer lugar se define un diccionario de configuración que centraliza los
hiperparámetros principales del experimento, seguido de las funciones necesarias para la
carga y preprocesado de CIFAR-10.

```python
# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    "batch_size": 128,
    "num_epochs": 1,
    "learning_rate": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 10,
}
```

```python
# ============================================================
# DATOS
# ============================================================

def get_dataloaders(batch_size: int = 128):
    """Prepara CIFAR-10 con data augmentation y normalización estándar."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader
```

### Módulos de atención usados en la ResNet simplificada

Para facilitar la integración en bloques residuales estándar se emplean versiones
simplificadas de SE y CBAM. Estas versiones mantienen la esencia de los mecanismos de
atención, pero se ajustan a una implementación compacta adecuada para su reutilización
sistemática en la arquitectura.

```python
# ============================================================
# ATTENTION MODULES (versión integrada)
# ============================================================

class SESimple(nn.Module):
    """Bloque SE simplificado: Atención de canal mediante pooling global + FC."""
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAMSimple(nn.Module):
    """CBAM simplificado: Atención de canal + atención espacial."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()

        # Atención de canal
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        )

        # Atención espacial
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Atención de canal
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Atención espacial
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_att

        return x
```

### Bloques residuales con atención opcional

La arquitectura residual se construye a partir de bloques básicos que contienen dos
convoluciones de $3 \times 3$ con normalización por lotes y una conexión de atajo
(_shortcut_). Sobre este esquema se superpone, de forma opcional, uno de los módulos de
atención descritos anteriormente. La selección del tipo de atención se controla mediante
un parámetro de tipo cadena.

```python
# ============================================================
# RESIDUAL BLOCKS
# ============================================================

class ResidualBlock(nn.Module):
    """Bloque residual básico con atención opcional (None, 'se' o 'cbam')."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        attention: str | None = None
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if attention == "se":
            self.attention = SESimple(out_channels)
        elif attention == "cbam":
            self.attention = CBAMSimple(out_channels)
        else:
            self.attention = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.attention is not None:
            out = self.attention(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

### Arquitectura ResNet-18 con atención opcional

Sobre los bloques residuales anteriores se construye una variante simplificada de
ResNet-18 adaptada a CIFAR-10, en la que se mantiene la estructura general de cuatro
grupos de bloques con incrementos progresivos en el número de canales. La atención se
incorpora de manera homogénea en todos los bloques de la red.

```python
# ============================================================
# RESNET18
# ============================================================

class ResNet18(nn.Module):
    """ResNet-18 simplificada para CIFAR-10 con atención opcional."""

    def __init__(self, num_classes: int = 10, attention: str | None = None) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, attention=attention)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, attention=attention)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, attention=attention)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, attention=attention)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        attention: str | None
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride, attention)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, attention))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### Entrenamiento y evaluación

El procedimiento de entrenamiento y evaluación se estructura en funciones diferenciadas.
La función `train_epoch` realiza un recorrido completo por el conjunto de entrenamiento,
mientras que `evaluate` calcula la pérdida y la precisión sobre un conjunto de validación
o prueba. Sobre estas funciones se construye `train_model`, que orquesta el proceso a lo
largo de múltiples épocas, aplica un plan de decremento de la tasa de aprendizaje
(_learning rate scheduler_) y registra la evolución de la precisión.

```python
# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total
```

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: str,
    name: str
):
    """Entrena un modelo y devuelve el historial de precisión."""
    print(f"\nEntrenando modelo: {name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 15],
        gamma=0.1
    )

    history = {"train_acc": [], "test_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    return history
```

### Pipeline principal y comparación de mecanismos de atención

El pipeline principal inicializa los cargadores de datos, define las tres variantes de
modelo (ResNet-18 base, SE-ResNet-18 y CBAM-ResNet-18) y entrena cada una de ellas
durante un número determinado de épocas. Posteriormente, representa gráficamente la
evolución de la precisión en el conjunto de prueba y muestra un resumen de los resultados
finales.

```python
# ============================================================
# MAIN PIPELINE
# ============================================================
def main() -> None:
    print("=" * 70)
    print("COMPARACIÓN DE MECANISMOS DE ATENCIÓN EN RESNET-18 (CIFAR-10)")
    print("=" * 70)

    train_loader, test_loader = get_dataloaders(CONFIG["batch_size"])
    device = CONFIG["device"]

    models = {
        "ResNet18": ResNet18(num_classes=CONFIG["num_classes"], attention=None),
        "SE-ResNet18": ResNet18(num_classes=CONFIG["num_classes"], attention="se"),
        "CBAM-ResNet18": ResNet18(num_classes=CONFIG["num_classes"], attention="cbam")
    }

    histories: dict[str, dict[str, list[float]]] = {}

    for name, model in models.items():
        histories[name] = train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=CONFIG["num_epochs"],
            device=device,
            name=name
        )

    # Representación gráfica de la precisión en test
    plt.figure(figsize=(10, 5))
    for name, history in histories.items():
        plt.plot(history["test_acc"], label=name, linewidth=2)

    plt.xlabel("Época")
    plt.ylabel("Accuracy en test (%)")
    plt.title("Comparación de mecanismos de atención en CIFAR-10")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("attention_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 70)
    print("RESULTADOS FINALES")
    print("=" * 70)
    for name, history in histories.items():
        final = history["test_acc"][-1]
        best = max(history["test_acc"])
        print(f"{name:20s}: {final:.2f}% (mejor: {best:.2f}%)")


if __name__ == "__main__":
    main()
```

Esta implementación permite cuantificar empíricamente el efecto de añadir atención de
canal (SE) o atención combinada canal-espacial (CBAM) a una arquitectura convolucional
profunda estándar. Aunque los resultados concretos dependen de los detalles de
entrenamiento (número de épocas, tasa de aprendizaje, regularización y otros factores),
se observa de forma recurrente que SE y CBAM mejoran la precisión frente a la ResNet
base, con un incremento de coste computacional moderado, especialmente en el caso de SE.
CBAM tiende a mostrar ventajas adicionales en escenarios donde la estructura espacial y
la localización de regiones relevantes desempeñan un papel destacado en la discriminación
entre clases, como en tareas de detección, segmentación o clasificación de escenas
complejas.
