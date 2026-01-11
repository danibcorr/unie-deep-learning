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
Attention Module_ (CBAM). 

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
