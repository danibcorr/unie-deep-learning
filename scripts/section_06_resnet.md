# ResNet

## Introducción teórica

### ResNet: Conexiones residuales y redes extremadamente profundas

La arquitectura **ResNet** (_Residual Network_), presentada por Kaiming He y
colaboradores en Microsoft Research en 2015, marca un punto de inflexión en el diseño de
redes neuronales profundas para visión por computador. Mientras que LeNet-5 inaugura el
uso sistemático de convoluciones y VGG consolida la profundidad como factor clave de
rendimiento, ResNet introduce un mecanismo estructural que permite entrenar redes
extremadamente profundas (de más de cien capas) de forma estable y eficaz.

ResNet adquiere gran relevancia al ganar el desafío **ImageNet 2015** con una variante de
**152 capas**, una profundidad que hasta entonces se consideraba prácticamente
inabordable desde el punto de vista del entrenamiento, debido a problemas de optimización
y estabilidad numérica.

## El problema de la degradación en redes profundas

Antes de ResNet, resultaba razonable suponer que incrementar el número de capas debía
conducir, al menos en principio, a modelos con mayor capacidad de representación y mejor
rendimiento. Sin embargo, los estudios empíricos mostraban que, a partir de un cierto
umbral (en torno a veinte o treinta capas), añadir más profundidad no solo dejaba de
mejorar el rendimiento, sino que **lo empeoraba**, incluso sobre el propio conjunto de
entrenamiento.

Este fenómeno se conoce como **degradación**. No se trata de un simple efecto de
_overfitting_, ya que el error aumenta también en entrenamiento, sino de una dificultad
estructural de optimización: en redes muy profundas, la señal de error que se propaga
hacia atrás tiende a atenuarse o volverse numéricamente inestable. Las capas cercanas a
la entrada reciben gradientes muy pequeños o ruidosos, de modo que apenas se actualizan,
lo que impide aprovechar la capacidad potencial del modelo.

### Evidencia experimental de la degradación

Los experimentos previos a ResNet muestran que una red de 56 capas puede presentar un
**error de entrenamiento mayor** que una red de 20 capas con arquitectura comparable.
Esta observación contradice una expectativa teórica básica: un modelo más profundo
debería ser capaz, como mínimo, de reproducir el rendimiento de uno más superficial, por
simple inclusión de hipótesis (bastaría con que algunas capas implementaran la función
identidad).

El hecho de que el error aumente incluso sobre el conjunto de entrenamiento pone de
manifiesto que el problema no reside en la capacidad de representación, sino en la
**dificultad de optimizar** redes muy profundas utilizando los mecanismos estándar de
entrenamiento basados en gradiente.

## Conexiones residuales y bloques de atajo

La innovación central de ResNet consiste en la introducción de **conexiones de atajo**
(_skip connections_) que dan lugar al **bloque residual**. La idea es conceptualmente
sencilla: en lugar de forzar a cada conjunto de capas a aprender una transformación
completa, la arquitectura permite que dicho conjunto aprenda solo la **diferencia** (o
residuo) entre la entrada y la salida deseada.

### Formulación matemática del bloque residual

En una red convencional, una capa o conjunto de capas recibe una entrada $x$ y aprende
una función $F(x)$. La salida del bloque es simplemente $F(x)$.

En ResNet, la salida de un bloque residual se define como $y = F(x) + x, $ donde:

- $x$ es la entrada al bloque residual.
- $F(x)$ es la transformación aprendida por las capas internas (convoluciones,
  normalización, activaciones).
- $y$ es la salida del bloque, obtenida como suma entre la entrada original y el residuo
  $F(x)$.

Esta suma crea una **ruta de identidad explícita**, a modo de “autopista” que atraviesa
la red, por la cual la información puede fluir sin ser modificada, en paralelo a las
transformaciones no lineales convencionales.

### Ventajas del aprendizaje residual

El aprendizaje residual aporta varias ventajas fundamentales.

En primer lugar, **facilita la optimización**. Si la transformación óptima en cierto
bloque es cercana a la identidad, resulta más sencillo para la red aprender una función
residual con $F(x) \approx 0$ que aprender directamente una transformación completa
$H(x) \approx x$ desde cero. El espacio de funciones residuales suele estar “más cerca”
del origen y, por tanto, es más accesible a los métodos de optimización basados en
gradiente.

En segundo lugar, la **propagación del gradiente** mejora de forma notable. Durante la
retropropagación, el gradiente de la pérdida $L$ con respecto a la entrada $x$ de un
bloque residual satisface, de forma simplificada,
$\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y} \left( \frac{\partial F}{\partial x} + I \right), $ donde
$I$
es la matriz identidad. Esta expresión garantiza que, aun en el caso extremo en que
$\frac{\partial
F}{\partial x}$ tienda a cero, existe siempre un camino directo para el gradiente a
través del término identidad $I$. En la práctica, esto evita que la señal se atenúe
completamente y contribuye a estabilizar el entrenamiento en redes muy profundas.

En tercer lugar, las conexiones residuales introducen una forma de **profundidad
adaptativa**. Si un bloque no resulta necesario para la tarea, puede aproximar
$F(x)
\approx 0$ y comportarse de hecho como una transformación identidad, de modo que
$y
\approx x$. La red conserva así la capacidad de “anular” bloques que no aportan
mejoras, sin comprometer el flujo de información.

En conjunto, estos mecanismos hacen posible entrenar redes con más de cien capas sin
sufrir la degradación severa que afectaba a arquitecturas previas. La señal de entrada
puede atravesar la red sin distorsionarse, y el gradiente dispone de rutas alternativas
que mitigan el desvanecimiento.

## Variantes arquitectónicas de ResNet

La familia ResNet incluye varias configuraciones que difieren principalmente en su
profundidad y en el tipo de bloque residual utilizado. De manera sintética:

|     Modelo | Capas | Parámetros | Bloques por etapa | Tipo de bloque |
| ---------: | ----: | ---------: | ----------------- | -------------- |
|  ResNet-18 |    18 |      ~11 M | [2, 2, 2, 2]      | Básico         |
|  ResNet-34 |    34 |      ~21 M | [3, 4, 6, 3]      | Básico         |
|  ResNet-50 |    50 |      ~25 M | [3, 4, 6, 3]      | Bottleneck     |
| ResNet-101 |   101 |      ~44 M | [3, 4, 23, 3]     | Bottleneck     |
| ResNet-152 |   152 |      ~60 M | [3, 8, 36, 3]     | Bottleneck     |

### Bloque básico frente a bloque _bottleneck_

En las versiones menos profundas, como **ResNet-18** y **ResNet-34**, se emplea el
**bloque básico**, con la estructura:

```text
x → [Conv 3×3] → [BN] → [ReLU] → [Conv 3×3] → [BN] → (+) → [ReLU]
↓______________________________________________________________|
```

Este bloque consta de dos convoluciones $3 \times 3$ seguidas de normalización por lotes
y activación ReLU, y una suma residual con la rama de identidad. El número de canales de
salida coincide con el de entrada (factor de expansión 1).

En las variantes más profundas, como **ResNet-50**, **ResNet-101** y **ResNet-152**, se
recurre al **bloque _bottleneck_** (cuello de botella), cuyo objetivo es reducir el coste
computacional manteniendo la capacidad representacional. Su estructura es:

```text
x → [Conv 1×1] → [BN] → [ReLU]
  → [Conv 3×3] → [BN] → [ReLU]
  → [Conv 1×1] → [BN] → (+) → [ReLU]
↓______________________________________________________________|
```

Este bloque combina tres convoluciones consecutivas. La primera, de tamaño $1 \times
1$,
reduce la dimensionalidad de los canales (por ejemplo, de 256 a 64 canales). La segunda,
de tamaño $3 \times 3$, realiza el procesamiento principal sobre un número reducido de
canales. La tercera, nuevamente de $1 \times 1$, restaura la dimensionalidad original
(por ejemplo, de 64 a 256 canales). La expansión típica es 4: los canales de salida son
cuatro veces los canales intermedios.

### Análisis de eficiencia computacional

La ventaja del diseño _bottleneck_ se aprecia al comparar el número de parámetros de
configuraciones equivalentes.

Si se consideran dos capas convolucionales $3 \times 3$ con 256 canales de entrada y
salida, el número total de parámetros es $\text{Parámetros} = 2 \times (256 \times 3
\times 3 \times 256) = 1\,179\,648. $

En cambio, para un bloque _bottleneck_ con patrón $256 \rightarrow 64 \rightarrow 256$:

Primera convolución $1 \times 1$: $256 \times 1 \times 1 \times 64 = 16\,384. $

Convolución central $3 \times 3$: $64 \times 3 \times 3 \times 64 = 36\,864. $

Última convolución $1 \times 1$: $64 \times 1 \times 1 \times 256 = 16\,384. $

El bloque completo contiene, aproximadamente,
$16\,384 + 36\,864 + 16\,384 = 69\,632 $
parámetros, lo que supone una **reducción cercana al 94 %** respecto a las dos capas $3
\times 3$
con 256 canales. El uso de capas $1 \times 1$ para reducir y restaurar dimensionalidad
permite mantener la capacidad representacional a la vez que se abarata de forma drástica
el cómputo.

## ResNet en perspectiva histórica

La comparación entre LeNet-5, VGG-16 y ResNet-50 ilustra la evolución en el diseño de
arquitecturas convolucionales profundas:

| Característica         | LeNet-5 (1998)      | VGG-16 (2014)        | ResNet-50 (2015)           |
| ---------------------- | ------------------- | -------------------- | -------------------------- |
| Capas                  | 7                   | 16                   | 50                         |
| Parámetros             | ~60 K               | ~138 M               | ~25 M                      |
| Innovación clave       | Convoluciones       | Profundidad uniforme | Skip connections           |
| Filtros principales    | $5 \times 5$        | $3 \times 3$         | $1 \times 1$, $3 \times 3$ |
| Problema abordado      | Dígitos manuscritos | Visión compleja      | Redes muy profundas        |
| Top-5 error (ImageNet) | —                   | ~7.3 %               | ~3.6 %                     |

LeNet-5 introduce la convolución y el _pooling_ como mecanismos esenciales para explotar
la estructura espacial de las imágenes. VGG-16 incrementa la profundidad hasta dieciséis
capas mediante una arquitectura homogénea basada en filtros $3 \times 3$, pero a costa de
un número de parámetros muy elevado (del orden de 138 millones). ResNet-50, por su parte,
alcanza cincuenta capas con unos veinticinco millones de parámetros, gracias al uso
sistemático de bloques _bottleneck_ y conexiones residuales, logrando una combinación muy
favorable de profundidad, estabilidad y eficiencia.

## Impacto e importancia actual de ResNet

En la actualidad, ResNet se considera una arquitectura de referencia tanto en el ámbito
académico como en aplicaciones industriales. El equilibrio entre profundidad, estabilidad
de entrenamiento y eficiencia la convierte en base de numerosos sistemas de
reconocimiento facial, conducción autónoma, diagnóstico médico asistido por imagen y
análisis visual a gran escala en distintos dominios.

Entre sus ventajas destacan el uso eficiente de parámetros (por ejemplo, ResNet-50 emplea
aproximadamente cinco veces menos parámetros que VGG-16), la capacidad de entrenar redes
con más de cien capas sin degradación severa del gradiente, su idoneidad como estructura
base para aprendizaje por transferencia, su estabilidad numérica durante el entrenamiento
y su versatilidad, que ha inspirado variantes en visión, procesamiento del lenguaje
natural y otras modalidades.

Más allá de resolver un problema técnico concreto, ResNet redefine el diseño de
arquitecturas profundas al incorporar explícitamente caminos de identidad que facilitan
el flujo de información y gradientes a través de la red.

## Implementación práctica de ResNet para CIFAR-10

A continuación se presenta una implementación completa y funcional de ResNet (ResNet-18,
ResNet-34 y ResNet-50) para el conjunto de datos **CIFAR-10** utilizando PyTorch. El
código está organizado para poder convertirse directamente en un cuaderno de Jupyter y
ejecutarse de forma secuencial, desde la carga de datos hasta el entrenamiento, la
evaluación y la visualización de resultados.

### Importación de bibliotecas

Se importan en primer lugar los módulos necesarios para la definición de la arquitectura,
la gestión de datos, el entrenamiento y la visualización.

```python
# Bibliotecas estándar
from typing import Any, List, Type, Union
import time

# Bibliotecas de terceros
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm

print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
```

### Configuración global de hiperparámetros

Se definen las constantes y los hiperparámetros que se emplean a lo largo del
experimento.

```python
# Configuración global
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 100
LEARNING_RATE: float = 0.1
WEIGHT_DECAY: float = 1e-4
MOMENTUM: float = 0.9
NUM_CLASSES: int = 10
INPUT_SIZE: int = 32

# Nombres de las clases de CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Configuración establecida:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Épocas: {NUM_EPOCHS}")
print(f"  Learning rate inicial: {LEARNING_RATE}")
print(f"  Momentum: {MOMENTUM}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Número de clases: {NUM_CLASSES}")
```

### Función auxiliar de visualización

La función `show_images` permite visualizar imágenes de CIFAR-10 junto con sus etiquetas
reales y, opcionalmente, las predicciones del modelo.

```python
def show_images(images, labels, predictions=None, classes=CIFAR10_CLASSES):
    """
    Visualiza un conjunto de imágenes con sus etiquetas y predicciones.

    Args:
        images: Tensor de imágenes [N, C, H, W].
        labels: Tensor de etiquetas [N].
        predictions: Tensor opcional de predicciones [N].
        classes: Lista de nombres de clases.
    """
    n_images = min(len(images), 8)
    fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 3))

    if n_images == 1:
        axes = [axes]

    for idx in range(n_images):
        img = images[idx]
        label = labels[idx]
        ax = axes[idx]

        # Desnormalizar imagen (asumiendo normalización estándar)
        img = img / 2 + 0.5
        img = img.numpy().transpose((1, 2, 0))

        ax.imshow(img)

        title = f"Real: {classes[label]}"
        if predictions is not None:
            pred = predictions[idx]
            color = "green" if pred == label else "red"
            title += f"\nPred: {classes[pred]}"
            ax.set_title(title, fontsize=9, color=color, fontweight="bold")
        else:
            ax.set_title(title, fontsize=9, fontweight="bold")

        ax.axis("off")

    plt.tight_layout()
    plt.show()

print("Función de visualización definida correctamente")
```

### Preparación del dataset CIFAR-10

Se carga CIFAR-10 y se definen las transformaciones de preprocesado y _data augmentation_
para entrenamiento y validación.

```python
# Estadísticas de normalización de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Transformaciones con data augmentation para entrenamiento
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Transformaciones para validación/prueba
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

print("Descargando dataset CIFAR-10...")

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

print("\nEstadísticas del dataset:")
print(f"  Muestras de entrenamiento: {len(train_dataset):,}")
print(f"  Muestras de prueba: {len(test_dataset):,}")
print(f"  Número de clases: {len(train_dataset.classes)}")
print("  Tamaño de imagen: 32×32 píxeles (RGB)")
```

El _data augmentation_ introduce variaciones de encuadre y simetría horizontal que ayudan
a mejorar la generalización. La normalización con la media y desviación estándar de
CIFAR-10 centra y escala cada canal, lo que suele favorecer una convergencia más estable.

### Creación de DataLoaders

Se definen los `DataLoader` de entrenamiento y prueba, configurando el número de procesos
de carga y otras opciones orientadas al rendimiento.

```python
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

print("DataLoaders configurados:")
print(f"  Batches de entrenamiento: {len(train_dataloader)}")
print(f"  Batches de prueba: {len(test_dataloader)}")
```

### Exploración visual del dataset

Antes de entrenar el modelo, resulta útil inspeccionar un lote de imágenes para verificar
la correcta configuración de la carga y el preprocesamiento.

```python
# Obtener un batch de entrenamiento
data_iter = iter(train_dataloader)
train_images, train_labels = next(data_iter)

print("\nDimensiones de un batch:")
print(f"  Imágenes: {train_images.shape}")
print(f"  Etiquetas: {train_labels.shape}")

print("\nVisualizando primeras 8 muestras...")
show_images(train_images[:8], train_labels[:8])
```

## Implementación de ResNet

Se implementan los bloques residuales básico y _bottleneck_, así como la clase principal
`ResNet`, adaptada al tamaño de imagen de CIFAR-10.

```python
class BasicBlock(nn.Module):
    """
    Bloque residual básico para ResNet-18 y ResNet-34.

    Estructura:
        x → [Conv 3×3] → [BN] → [ReLU]
          → [Conv 3×3] → [BN] → (+) → [ReLU]
        ↓__________________________________________________|

    Expansión: 1 (número de canales de salida = número de canales de entrada).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super().__init__()

        # Primera convolución 3×3
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Segunda convolución 3×3
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Rama de atajo (ajuste dimensional si es necesario)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out
```

```python
class Bottleneck(nn.Module):
    """
    Bloque bottleneck para ResNet-50, ResNet-101 y ResNet-152.

    Estructura:
        x → [Conv 1×1] → [BN] → [ReLU]
          → [Conv 3×3] → [BN] → [ReLU]
          → [Conv 1×1] → [BN] → (+) → [ReLU]
        ↓__________________________________________________|

    Expansión: 4 (canales de salida = 4 × canales intermedios).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super().__init__()

        # Conv 1×1 para reducir dimensionalidad
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv 3×3 para procesamiento principal
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Conv 1×1 para restaurar dimensionalidad
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out
```

```python
class ResNet(nn.Module):
    """
    Implementación de ResNet adaptada para CIFAR-10.

    Diferencias respecto a la versión original para ImageNet:
      - Primera capa: Conv 3×3 en lugar de Conv 7×7.
      - No se utiliza MaxPooling inicial (las imágenes son 32×32).
      - La capa final de clasificación se ajusta a CIFAR-10.

    Args:
        block: Tipo de bloque (BasicBlock o Bottleneck).
        layers: Lista con el número de bloques por etapa.
        num_classes: Número de clases de salida.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10
    ) -> None:
        super().__init__()

        self.in_channels = 64

        # Capa inicial adaptada a CIFAR-10 (32×32)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Cuatro etapas de bloques residuales
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Pooling global y clasificador final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Inicialización de pesos
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Construye una etapa de bloques residuales.

        Args:
            block: Tipo de bloque residual.
            out_channels: Número de canales de salida.
            num_blocks: Número de bloques en la etapa.
            stride: Stride del primer bloque (downsampling).

        Returns:
            nn.Sequential con los bloques de la etapa.
        """
        downsample = None

        # Ajuste de dimensionalidad en la rama de atajo
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """
        Inicializa pesos utilizando la inicialización de He (Kaiming).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de ResNet.

        Args:
            x: Tensor de entrada [B, 3, 32, 32].

        Returns:
            Logits de clasificación [B, num_classes].
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32×32 → 32×32
        x = self.layer2(x)  # 32×32 → 16×16
        x = self.layer3(x)  # 16×16 →  8×8
        x = self.layer4(x)  #  8×8 →  4×4

        x = self.avgpool(x)  # 4×4 → 1×1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae características antes de la capa de clasificación.

        Útil para visualización de embeddings y transfer learning.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
```

Se definen funciones de conveniencia para crear las variantes principales:

```python
def resnet18(num_classes: int = 10) -> ResNet:
    """Construye una ResNet-18 para el número de clases indicado."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 10) -> ResNet:
    """Construye una ResNet-34 para el número de clases indicado."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 10) -> ResNet:
    """Construye una ResNet-50 para el número de clases indicado."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


print("Arquitectura ResNet definida correctamente")
```

En esta implementación se separan claramente los componentes fundamentales: `BasicBlock`
para ResNet-18/34, `Bottleneck` para ResNet-50/101/152 y la clase `ResNet`, que ensambla
las etapas, gestiona el _downsampling_ y aplica _global average pooling_ antes de la
clasificación final.

### Instanciación y análisis del modelo

Se crea una instancia de ResNet-18 adaptada a CIFAR-10 y se analiza su estructura y
número de parámetros mediante `torchinfo.summary`.

```python
# Crear ResNet-18
model = resnet18(num_classes=NUM_CLASSES)

# Determinar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Dispositivo utilizado: {device}")
print(f"\n{'='*70}")
print("RESUMEN DE LA ARQUITECTURA ResNet-18")
print(f"{'='*70}\n")

summary(model, input_size=(BATCH_SIZE, 3, 32, 32), device=str(device))

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

print(f"\n{'='*70}")
print("ANÁLISIS DE PARÁMETROS POR COMPONENTE")
print(f"{'='*70}")
print(f"  Conv inicial:     {count_parameters(model.conv1):>12,} parámetros")
print(f"  Layer 1 (64 c.):  {count_parameters(model.layer1):>12,} parámetros")
print(f"  Layer 2 (128 c.): {count_parameters(model.layer2):>12,} parámetros")
print(f"  Layer 3 (256 c.): {count_parameters(model.layer3):>12,} parámetros")
print(f"  Layer 4 (512 c.): {count_parameters(model.layer4):>12,} parámetros")
print(f"  Clasificador FC:  {count_parameters(model.fc):>12,} parámetros")
print(f"  {'-'*66}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  TOTAL:            {total_params:>12,} parámetros")
print(f"  Entrenables:      {trainable_params:>12,} parámetros")
print(f"  Memoria (float32): {total_params * 4 / (1024**2):>10.2f} MB")
```

### Configuración del entrenamiento

Se emplea un optimizador SGD con _momentum_ de Nesterov y una política de programación
del _learning rate_ mediante `MultiStepLR`, configuración habitual para ResNet en
CIFAR-10.

```python
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print(f"{'='*70}")
print(f"  Épocas: {NUM_EPOCHS}")
print(f"  Learning rate inicial: {LEARNING_RATE}")
print(f"  Momentum: {MOMENTUM}")
print(f"  Weight decay (L2): {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"{'='*70}\n")

# Optimizador: SGD con momentum de Nesterov
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True
)

# Scheduler: MultiStepLR
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[60, 80],
    gamma=0.1,
    verbose=True
)

# Función de pérdida
loss_function = nn.CrossEntropyLoss()

print("Optimizador: SGD con Nesterov momentum")
print("  El momentum de Nesterov incorpora un 'look-ahead' en la dirección de descenso")
print("\nScheduler: MultiStepLR")
print("  Reduce el learning rate ×0.1 en las épocas 60 y 80")
print("  Estrategia clásica para entrenar ResNet en CIFAR-10")
print("\nFunción de pérdida: CrossEntropyLoss")
```

En este contexto, el calendario típico de `MultiStepLR` es:

- Épocas 0–60: $\text{LR} = 0.1$ (fase de exploración).
- Épocas 60–80: $\text{LR} = 0.01$ (fase de refinamiento).
- Épocas 80–100: $\text{LR} = 0.001$ (fase de ajuste fino).

### Bucle de entrenamiento y validación

Se implementa el bucle de entrenamiento con seguimiento de métricas y guardado del mejor
modelo según la precisión en el conjunto de prueba.

```python
# Estructuras para almacenar métricas
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
learning_rates = []

# Variables para guardar el mejor modelo
best_test_acc = 0.0
best_epoch = 0

def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total

print("INICIANDO ENTRENAMIENTO\n")
print(f"{'='*70}\n")

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    # Fase de entrenamiento
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    train_loop = tqdm(
        train_dataloader,
        desc=f"Época {epoch + 1}/{NUM_EPOCHS} [TRAIN]",
        leave=False
    )

    for batch_image, batch_label in train_loop:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        optimizer.zero_grad()

        outputs = model(batch_image)
        loss = loss_function(outputs, batch_label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_correct, batch_total = calculate_accuracy(outputs, batch_label)
        correct += batch_correct
        total += batch_total

        train_loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    epoch_train_loss = running_loss / len(train_dataloader)
    epoch_train_acc = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Fase de validación
    model.eval()
    test_loss, correct_test, total_test = 0.0, 0, 0

    test_loop = tqdm(
        test_dataloader,
        desc=f"Época {epoch + 1}/{NUM_EPOCHS} [TEST]",
        leave=False
    )

    with torch.no_grad():
        for images, labels in test_loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            batch_correct, batch_total = calculate_accuracy(outputs, labels)
            correct_test += batch_correct
            total_test += batch_total

            test_loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_test / total_test:.2f}%"
            })

    epoch_test_loss = test_loss / len(test_dataloader)
    epoch_test_acc = 100 * correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)

    # Actualizar scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    # Guardar mejor modelo según precisión en test
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        best_epoch = epoch + 1
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_acc": best_test_acc,
        }, "resnet18_cifar10_best.pth")

    epoch_time = time.time() - epoch_start_time

    print(f"Época [{epoch + 1}/{NUM_EPOCHS}] - Tiempo: {epoch_time:.2f}s")
    print(f"  Train → Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
    print(f"  Test  → Loss: {epoch_test_loss:.4f} | Acc: {epoch_test_acc:.2f}%")
    print(f"  LR: {current_lr:.6f} | Mejor test acc: {best_test_acc:.2f}% (época {best_epoch})")
    print(f"  {'─'*66}\n")

total_time = time.time() - start_time

print(f"\n{'='*70}")
print("ENTRENAMIENTO COMPLETADO")
print(f"{'='*70}")
print(f"  Tiempo total: {total_time / 60:.2f} minutos")
print(f"  Tiempo promedio por época: {total_time / NUM_EPOCHS:.2f} segundos")
print(f"  Precisión final (test): {test_accuracies[-1]:.2f}%")
print(f"  Mejor precisión (test): {best_test_acc:.2f}% en época {best_epoch}")

# Guardar modelo final y métricas
torch.save({
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "test_losses": test_losses,
    "test_accuracies": test_accuracies,
    "best_test_acc": best_test_acc,
    "best_epoch": best_epoch,
}, "resnet18_cifar10_final.pth")

print("\nModelos guardados:")
print("  - resnet18_cifar10_best.pth (mejor modelo)")
print("  - resnet18_cifar10_final.pth (modelo final + métricas)")
```

En comparación con arquitecturas como VGG, el entrenamiento de ResNet sobre CIFAR-10
tiende a ser más estable y eficiente para profundidades similares, gracias a las
conexiones residuales y al uso moderado de parámetros.

### Visualización de métricas de entrenamiento

Se representa la evolución de la pérdida, la precisión y el _learning rate_, así como la
diferencia entre precisión de entrenamiento y prueba.

```python
epochs_range = range(1, NUM_EPOCHS + 1)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Pérdida
ax1.plot(epochs_range, train_losses, "o-", label="Train Loss",
         linewidth=2, markersize=3, alpha=0.7)
ax1.plot(epochs_range, test_losses, "s-", label="Test Loss",
         linewidth=2, markersize=3, alpha=0.7)
ax1.axvline(
    x=best_epoch,
    color="green",
    linestyle="--",
    alpha=0.5,
    label=f"Mejor época ({best_epoch})"
)
ax1.set_xlabel("Época", fontsize=12, fontweight="bold")
ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax1.set_title("Evolución de la Pérdida", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Precisión
ax2.plot(epochs_range, train_accuracies, "o-", label="Train Accuracy",
         linewidth=2, markersize=3, alpha=0.7)
ax2.plot(epochs_range, test_accuracies, "s-", label="Test Accuracy",
         linewidth=2, markersize=3, alpha=0.7)
ax2.axvline(
    x=best_epoch,
    color="green",
    linestyle="--",
    alpha=0.5,
    label=f"Mejor época ({best_epoch})"
)
ax2.axhline(
    y=best_test_acc,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Mejor acc: {best_test_acc:.2f}%"
)
ax2.set_xlabel("Época", fontsize=12, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax2.set_title("Evolución de la Precisión", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Learning rate
ax3.plot(epochs_range, learning_rates, "o-", color="red",
         linewidth=2, markersize=3, alpha=0.7)
ax3.set_xlabel("Época", fontsize=12, fontweight="bold")
ax3.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3)
ax3.axvline(x=60, color="orange", linestyle="--", alpha=0.5, label="LR decay")
ax3.axvline(x=80, color="orange", linestyle="--", alpha=0.5)
ax3.legend(fontsize=10)

# Diferencia train-test (gap)
gap = np.array(train_accuracies) - np.array(test_accuracies)
ax4.plot(epochs_range, gap, "o-", color="purple",
         linewidth=2, markersize=3, alpha=0.7)
ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax4.axhline(
    y=5,
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Umbral overfitting (5%)"
)
ax4.set_xlabel("Época", fontsize=12, fontweight="bold")
ax4.set_ylabel("Gap Train-Test (%)", fontsize=12, fontweight="bold")
ax4.set_title("Diferencia Train-Test Accuracy", fontsize=14, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resnet18_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nAnálisis de resultados:")
final_gap = train_accuracies[-1] - test_accuracies[-1]
print(f"  Overfitting detectado: {'SÍ' if final_gap > 10 else 'NO'}")
print(f"  Gap final train-test: {final_gap:.2f}%")
print(f"  Mejor época: {best_epoch}")
print(f"  Mejora desde época 1: {test_accuracies[-1] - test_accuracies[0]:.2f}%")
```

Valores moderados del _gap_ entre precisión de entrenamiento y de prueba indican un buen
equilibrio entre ajuste y generalización. Los descensos en la tasa de aprendizaje en las
épocas 60 y 80 suelen ir acompañados de una ligera reconfiguración del comportamiento de
la red, con mejoras posteriores en la precisión sobre el conjunto de prueba.

### Visualización de predicciones del modelo

Finalmente, se visualizan predicciones del modelo sobre el conjunto de prueba, incluyendo
ejemplos correctamente clasificados y algunos errores, lo que permite una inspección
cualitativa del comportamiento del modelo.

```python
print("\nVisualizando predicciones del mejor modelo...")

# Obtener un batch de test
data_iter = iter(test_dataloader)
test_images, test_labels = next(data_iter)

model.eval()
with torch.no_grad():
    test_images_device = test_images.to(device)
    outputs = model(test_images_device)
    _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu()

print("\nPrimeras 8 predicciones:")
show_images(test_images[:8], test_labels[:8], predictions[:8])

# Ejemplos de errores de clasificación
incorrect_indices = (predictions != test_labels).nonzero(as_tuple=True)[0]

if len(incorrect_indices) >= 8:
    print("\nEjemplos de predicciones incorrectas:")
    error_indices = incorrect_indices[:8]
    show_images(
        test_images[error_indices],
        test_labels[error_indices],
        predictions[error_indices]
    )
else:
    print(f"\nSolo {len(incorrect_indices)} errores en este batch")
```

```

```
