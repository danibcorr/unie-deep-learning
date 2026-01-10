# VGG

## Introducción teórica

### Profundidad y simplicidad en redes convolucionales

La arquitectura **VGG**, desarrollada por el _Visual Geometry Group_ de la Universidad de
Oxford y presentada en 2014, constituye un hito en la evolución del aprendizaje profundo
aplicado a la visión por computador. Mientras que LeNet-5 establece las bases
conceptuales de las redes neuronales convolucionales, VGG demuestra de manera sistemática
que el incremento de la **profundidad** de la red, combinado con una estructura
extremadamente simple y homogénea, conduce a mejoras significativas en el rendimiento.
Las variantes más influyentes de esta familia son **VGG-16** y **VGG-19**, denominadas
según el número de capas con parámetros entrenables que las componen.

El diseño de VGG se caracteriza por su simplicidad estructural: una secuencia de
convoluciones de tamaño fijo, separadas periódicamente por operaciones de _pooling_,
seguida de un bloque de capas totalmente conectadas. Esta filosofía minimalista contrasta
con arquitecturas posteriores más complejas, como Inception, y convierte a VGG en una
referencia didáctica fundamental para comprender redes convolucionales profundas. La
arquitectura demuestra que es posible obtener representaciones visuales altamente
discriminativas mediante la repetición sistemática de componentes básicos, sin necesidad
de introducir operaciones complejas o módulos especializados.

## Principios de diseño

El rasgo distintivo de VGG reside en su filosofía de diseño deliberadamente conservadora
y homogénea. A diferencia de arquitecturas previas que emplean filtros convolucionales de
distintos tamaños (por ejemplo, $5 \times 5$ o $7 \times 7$), VGG estandariza
estrictamente sus componentes. La arquitectura utiliza exclusivamente filtros
convolucionales de tamaño **$3 \times 3$** aplicados de forma repetida, e incrementa
progresivamente el número de filtros a medida que disminuye la resolución espacial de los
mapas de características, generalmente duplicando dicho número tras cada operación de
_max pooling_.

Además, VGG utiliza convoluciones con _padding_ adecuado para preservar la resolución
espacial dentro de cada bloque, y operaciones de _max pooling_ de **$2 \times 2$** con
paso (_stride_) 2 para reducir la dimensión espacial al final de cada bloque. Esta
combinación da lugar a una jerarquía de representaciones en la que las capas tempranas
capturan información local de bajo nivel, mientras que las capas profundas agrupan
contextos espaciales más amplios y codifican características de mayor abstracción.

### Justificación del uso de filtros pequeños

La elección de filtros convolucionales de tamaño **$3 \times 3$** responde a una
combinación de argumentos teóricos y prácticos. Aunque en LeNet-5 se utilizan filtros de
**$5 \times 5$**, los diseñadores de VGG demuestran que varias capas consecutivas de
**$3 \times 3$** pueden emular el campo receptivo de filtros de mayor tamaño, con menos
parámetros y un mayor número de no linealidades intermedias.

Desde el punto de vista del campo receptivo, una única convolución de tamaño $k \times k$
puede aproximarse mediante una secuencia de convoluciones de tamaño menor (por ejemplo,
$3 \times 3$), siempre que se mantenga un _padding_ adecuado. En particular, dos capas de
$3 \times 3$ tienen un campo receptivo efectivo equivalente a uno de $5 \times 5$, y tres
capas de $3 \times 3$ se aproximan a uno de $7 \times 7$. Esta descomposición resulta
ventajosa tanto en términos de eficiencia de parámetros como de capacidad expresiva del
modelo.

Para entender esta equivalencia, se considera el **campo receptivo efectivo** de una
neurona en una red convolucional. De forma intuitiva, el campo receptivo indica cuántos
píxeles de la imagen original influyen en la activación de una neurona en una capa
determinada. En el caso más simple, con _stride_ 1 y _padding_ apropiado, se cumple lo
siguiente:

- Una capa convolucional de tamaño $5 \times 5$ posee un campo receptivo de
  $5 \times 5 = 25$ píxeles.
- Dos capas consecutivas de tamaño $3 \times 3$ producen un campo receptivo efectivo de
  $5 \times 5$. La primera capa observa $3 \times 3$ píxeles de la imagen, y la segunda
  observa $3 \times 3$ neuronas de la capa anterior, cuyos campos receptivos se solapan
  de forma que, combinados, abarcan $5 \times 5$ píxeles de la imagen original.

Más en general, si se encadenan $L$ convoluciones de tamaño $3 \times 3$ con _stride_ 1 y
_padding_ 1, el tamaño del campo receptivo efectivo viene dado por:

$$
k_{\text{efectivo}} = 1 + 2L.
$$

De esta manera, dos capas de $3 \times 3$ ($L = 2$) producen $k_{\text{efectivo}} = 5$, y
tres capas ($L = 3$) producen $k_{\text{efectivo}} = 7$.

Desde el punto de vista del número de parámetros, la ventaja resulta evidente. Supóngase,
para simplificar el análisis, una misma dimensión de canal $C$ tanto de entrada como de
salida:

- Una capa con filtros de $5 \times 5$ contiene:

  $$
  \text{Parámetros} = C \times (5 \times 5 \times C) = 25 C^2.
  $$

- Dos capas consecutivas con filtros de $3 \times 3$ contienen:

  $$
  \text{Parámetros} = C \times (3 \times 3 \times C) + C \times (3 \times 3 \times C) = 18 C^2.
  $$

La sustitución de una capa de $5 \times 5$ por dos capas de $3 \times 3$ reduce el número
de parámetros de $25 C^2$ a $18 C^2$, lo que representa aproximadamente un **28 % de
reducción** de parámetros para un campo receptivo efectivo equivalente. Además, al
introducir una capa adicional se incorpora una función de activación no lineal intermedia
(por ejemplo, ReLU), lo que aumenta la capacidad expresiva de la red al permitir la
composición de transformaciones más complejas. En consecuencia, el uso sistemático de
filtros $3 \times 3$ representa un compromiso eficiente entre tamaño del campo receptivo,
coste computacional y capacidad de representación.

## Organización arquitectónica de VGG-16

La variante **VGG-16** recibe imágenes de entrada en color (RGB) de tamaño
**$224 \times 224$ píxeles** y se organiza en una secuencia jerárquica de cinco bloques
convolucionales, seguidos por un conjunto de capas totalmente conectadas encargadas de la
clasificación final. Cada bloque agrupa varias convoluciones $3 \times 3$ seguidas de una
operación de _max pooling_ $2 \times 2$.

En su configuración original para ImageNet, la arquitectura se organiza del modo
siguiente:

| Bloque   | Capas Conv | Filtros | Tamaño salida típico       | Pooling              |
| -------- | ---------: | ------: | -------------------------- | -------------------- |
| Bloque 1 |          2 |      64 | $112 \times 112 \times 64$ | MaxPool $2 \times 2$ |
| Bloque 2 |          2 |     128 | $56 \times 56 \times 128$  | MaxPool $2 \times 2$ |
| Bloque 3 |          3 |     256 | $28 \times 28 \times 256$  | MaxPool $2 \times 2$ |
| Bloque 4 |          3 |     512 | $14 \times 14 \times 512$  | MaxPool $2 \times 2$ |
| Bloque 5 |          3 |     512 | $7 \times 7 \times 512$    | MaxPool $2 \times 2$ |
| FC6      |          - |    4096 | 4096                       | -                    |
| FC7      |          - |    4096 | 4096                       | -                    |
| FC8      |          - |    1000 | 1000                       | Softmax              |

Los dos primeros bloques convolucionales están formados por dos capas de convolución cada
uno. El primer bloque emplea **64 filtros**, mientras que el segundo utiliza **128
filtros**, manteniendo siempre el tamaño $3 \times 3$. Al final de cada bloque se aplica
una operación de _max pooling_ de $2 \times 2$, cuya función es reducir a la mitad la
resolución espacial y concentrar la información más relevante.

Los bloques tercero, cuarto y quinto incrementan la profundidad efectiva de la red
mediante tres capas convolucionales consecutivas por bloque. El número de filtros
asciende a **256** en el tercer bloque y a **512** en los dos últimos. En estas etapas
profundas, la red aprende características altamente abstractas, tales como partes de
objetos, texturas complejas y configuraciones visuales de alto nivel, que resultan
cruciales para la discriminación entre clases.

Tras los bloques convolucionales, los mapas de características resultantes se transforman
en un vector unidimensional que alimenta las capas clasificadoras. El segmento final
consta de dos capas densas de **4096 neuronas** cada una, seguidas de una capa de salida
de **1000 neuronas**, correspondiente a las mil categorías del conjunto de datos
ImageNet. La activación **Softmax** permite interpretar la salida como una distribución
de probabilidad sobre las clases. A lo largo de toda la arquitectura se utiliza la
función de activación **ReLU**, que acelera el entrenamiento y contribuye a mitigar el
problema del desvanecimiento del gradiente.

El conjunto de estas decisiones de diseño —profundidad moderada, filtros pequeños,
estructura homogénea— sitúa a VGG-16 como un modelo de alto rendimiento en ImageNet,
aunque a costa de un número muy elevado de parámetros.

## Impacto, ventajas y limitaciones

VGG-16 obtiene el segundo puesto en el desafío ImageNet de 2014, pero su impacto
trasciende ampliamente la competición. La comunidad científica y técnica adopta esta
arquitectura como referencia debido a su diseño claro, regular y fácil de interpretar.
Esta claridad la convierte en herramienta fundamental tanto para la investigación como
para la docencia en redes convolucionales profundas, así como para el desarrollo de
numerosos trabajos posteriores en transferencia de aprendizaje y extracción de
características.

Entre sus **ventajas** destacan su arquitectura homogénea y modular, que facilita la
implementación y la experimentación; su excelente capacidad para aprender características
jerárquicas en imágenes; y su idoneidad para el **aprendizaje por transferencia**. Las
capas iniciales de VGG aprenden representaciones genéricas y robustas centradas en
bordes, texturas y patrones locales que pueden reutilizarse de forma eficaz en una amplia
variedad de tareas de visión por computador mediante la adaptación de las capas finales.

No obstante, VGG presenta también **limitaciones significativas**. El número muy elevado
de parámetros (del orden de 138 millones en VGG-16) implica un consumo de memoria
considerable (en torno a 500 MB en precisión de 32 bits) y un coste computacional alto
tanto en entrenamiento como en inferencia. Estas características hacen que la
arquitectura no resulte adecuada para dispositivos con recursos limitados y encarecen el
despliegue en producción a gran escala. Además, una parte sustancial de los parámetros se
concentra en las capas totalmente conectadas finales, lo que ha motivado que en
arquitecturas posteriores se sustituyan estas capas por mecanismos más eficientes como el
_global average pooling_.

Estas restricciones impulsan el desarrollo de arquitecturas posteriores, como Inception,
ResNet o MobileNet, orientadas a mantener o mejorar el rendimiento reduciendo el coste
computacional, facilitando el entrenamiento de redes más profundas y adaptándose a
entornos con recursos restringidos. A pesar de ello, VGG sigue siendo un modelo de
referencia clásico por su transparencia conceptual y por su capacidad para servir como
punto de partida en numerosas aplicaciones prácticas.

## Implementación práctica de VGG-16 en CIFAR-10

En esta sección se implementa una variante de **VGG-16** adaptada al conjunto de datos
**CIFAR-10**, que contiene imágenes en color de **$32 \times 32$ píxeles** pertenecientes
a 10 categorías. La arquitectura original, diseñada para ImageNet ($224 \times 224$), se
modifica para ajustarse al tamaño de entrada y al número de clases de CIFAR-10,
manteniendo la filosofía del diseño de VGG.

### Importación de bibliotecas

Se importan los módulos necesarios para implementar VGG-16 y entrenarla sobre CIFAR-10.
El objetivo es disponer de las herramientas de PyTorch para definir el modelo, gestionar
los datos, entrenar la red y analizar los resultados de forma sistemática.

```python
# Bibliotecas estándar
from typing import Any, List
import time

# Bibliotecas de terceros
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
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

Este bloque verifica la compatibilidad con GPU cuando está disponible y proporciona
información básica sobre el entorno de ejecución, lo que resulta útil para reproducir
experimentos y diagnosticar problemas de configuración.

### Configuración global de hiperparámetros

A continuación se definen constantes que se utilizan a lo largo de la implementación,
tales como el tamaño de _batch_, el número de épocas, la tasa de aprendizaje o el número
de clases. Centralizar estos parámetros facilita la experimentación y el ajuste del
modelo.

```python
# Configuración global
BATCH_SIZE: int = 128
NUM_EPOCHS: int = 1
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 5e-4
NUM_CLASSES: int = 10  # CIFAR-10 tiene 10 clases
INPUT_SIZE: int = 32   # CIFAR-10: imágenes 32×32

# Nombres de las clases de CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("Configuración establecida:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Épocas: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Número de clases: {NUM_CLASSES}")
```

Esta configuración proporciona un punto de partida razonable para entrenar VGG-16 en
CIFAR-10, estableciendo un compromiso entre rendimiento y coste computacional que puede
ajustarse según los recursos disponibles.

### Función auxiliar de visualización

La exploración visual de los datos ayuda a comprender mejor el problema y a verificar que
el preprocesamiento se aplica correctamente. Se define una función que permite visualizar
imágenes con sus etiquetas y, opcionalmente, con las predicciones del modelo.

```python
def show_images(images, labels, predictions=None, classes=CIFAR10_CLASSES):
    """
    Visualiza un conjunto de imágenes con sus etiquetas.

    Args:
        images: Tensor de imágenes [N, C, H, W]
        labels: Tensor de etiquetas [N]
        predictions: Tensor opcional de predicciones [N]
        classes: Lista de nombres de clases
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(2 * n_images, 3))

    if n_images == 1:
        axes = [axes]

    for idx, (img, label, ax) in enumerate(zip(images, labels, axes)):
        # Desnormalizar imagen para visualización
        img = img / 2 + 0.5  # Revertir normalización [-1, 1] -> [0, 1]
        img = img.numpy().transpose((1, 2, 0))

        ax.imshow(img)

        title = f"Real: {classes[label]}"
        if predictions is not None:
            pred = predictions[idx]
            color = "green" if pred == label else "red"
            title += f"\nPred: {classes[pred]}"
            ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        else:
            ax.set_title(title, fontsize=10, fontweight="bold")

        ax.axis("off")

    plt.tight_layout()
    plt.show()


print("Función de visualización definida correctamente")
```

Esta función se emplea posteriormente para inspeccionar tanto muestras del conjunto de
datos como predicciones correctas e incorrectas del modelo, proporcionando una
herramienta de diagnóstico visual esencial.

### Preparación del dataset CIFAR-10

Se cargan los datos de CIFAR-10 y se definen las transformaciones de _data augmentation_
y normalización aplicadas a las imágenes. CIFAR-10 contiene 60 000 imágenes en color de
$32 \times 32$ píxeles distribuidas en 10 clases, y presenta mayor variabilidad y
complejidad que MNIST debido a la diversidad de objetos, fondos y condiciones de captura.

```python
# Transformaciones con Data Augmentation para entrenamiento
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Recorte aleatorio con padding
    transforms.RandomHorizontalFlip(),     # Volteo horizontal aleatorio
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),  # Media RGB de CIFAR-10
                         (0.2470, 0.2435, 0.2616))  # Std RGB de CIFAR-10
])

# Transformaciones para validación (sin augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

print("Descargando dataset CIFAR-10...")

# Conjunto de entrenamiento
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

# Conjunto de prueba
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
print(f"  Clases: {', '.join(train_dataset.classes)}")
print("  Tamaño de imagen: 32×32 píxeles (RGB)")
```

El _data augmentation_ se introduce para aumentar la capacidad de generalización del
modelo. El `RandomCrop` con _padding_ simula variaciones de encuadre y posición del
objeto, mientras que el `RandomHorizontalFlip` incrementa la robustez frente a simetrías
horizontales. Ambos mecanismos reducen el sobreajuste al generar versiones ligeramente
diferentes de cada imagen en cada época, efectivamente ampliando el conjunto de
entrenamiento sin requerir datos adicionales.

La normalización se realiza utilizando la media y la desviación estándar del conjunto
completo de CIFAR-10 para cada canal RGB:

$$
\mu = (0.4914, 0.4822, 0.4465), \quad
\sigma = (0.2470, 0.2435, 0.2616).
$$

Esta operación centra y escala los valores de cada canal mediante la transformación:

$$
x_{\text{normalizado}} = \frac{x - \mu}{\sigma},
$$

lo que facilita la optimización y estabiliza el entrenamiento al mejorar el
condicionamiento numérico de los gradientes.

### Creación de DataLoaders

Una vez definidos los conjuntos de datos, se construyen los `DataLoader` que gestionan la
iteración por lotes durante el entrenamiento y la evaluación, incorporando optimizaciones
para acelerar la carga de datos.

```python
print("Configuración de DataLoaders:")
print(f"  Batch size: {BATCH_SIZE}")

# DataLoader de entrenamiento
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True
)

# DataLoader de prueba
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True
)

print(f"  Batches de entrenamiento: {len(train_dataloader)}")
print(f"  Batches de prueba: {len(test_dataloader)}")
```

Se aplican optimizaciones habituales en PyTorch. El parámetro `num_workers=4` permite
cargar datos en paralelo mediante procesos auxiliares, aprovechando múltiples núcleos de
CPU. El uso de `pin_memory=True` mejora la velocidad de transferencia de datos hacia la
GPU al utilizar memoria _pinned_ (no pageable). Finalmente, `persistent_workers=True`
evita recrear los procesos de carga en cada época, reduciendo el sobrecoste de
inicialización y acelerando el pipeline de datos.

### Exploración visual inicial del dataset

Antes de definir el modelo, resulta útil examinar algunas imágenes del conjunto de
entrenamiento y verificar las dimensiones de los tensores y el efecto de la
normalización.

```python
# Obtener un batch de datos
data_iter = iter(train_dataloader)
train_images, train_labels = next(data_iter)

print("Dimensiones de un batch:")
print(f"  Imágenes: {train_images.shape}")
print(f"  Etiquetas: {train_labels.shape}")
print(f"\n  Interpretación: {BATCH_SIZE} imágenes RGB de 32×32 píxeles")

# Visualizar primeros 8 ejemplos
print("\nVisualizando primeras 8 muestras...")
show_images(train_images[:8], train_labels[:8])

# Estadísticas de las imágenes normalizadas
print("\nEstadísticas después de normalización:")
print(f"  Valor mínimo: {train_images.min():.3f}")
print(f"  Valor máximo: {train_images.max():.3f}")
print("  Media por canal:")
for i, channel in enumerate(["R", "G", "B"]):
    print(f"    {channel}: {train_images[:, i, :, :].mean():.3f}")
```

Este análisis permite verificar que los datos se cargan correctamente, que el
preprocesamiento se aplica de forma adecuada y que las transformaciones de _data
augmentation_ producen variaciones razonables sin distorsionar excesivamente las
imágenes.

### Definición de la arquitectura VGG-16 para CIFAR-10

Se implementa a continuación una versión de VGG-16 adaptada a imágenes de
**$32 \times 32$** píxeles. La arquitectura original para ImageNet está pensada para
**$224 \times 224$**, por lo que se requieren ajustes en las capas finales para adaptarse
al tamaño reducido de entrada y al menor número de clases.

```python
class VGG16(nn.Module):
    """
    Implementación de VGG-16 adaptada para CIFAR-10 (32×32 píxeles).

    Arquitectura:
    - 5 bloques convolucionales con configuración [64, 128, 256, 512, 512]
    - Todos los filtros son de 3×3
    - MaxPooling 2×2 después de cada bloque
    - 3 capas fully connected al final
    - BatchNorm para estabilizar entrenamiento
    - Dropout para regularización
    """

    def __init__(self, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes

        # Bloque 1: 2 capas conv con 64 filtros
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32×32 -> 16×16
        )

        # Bloque 2: 2 capas conv con 128 filtros
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16×16 -> 8×8
        )

        # Bloque 3: 3 capas conv con 256 filtros
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8×8 -> 4×4
        )

        # Bloque 4: 3 capas conv con 512 filtros
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4×4 -> 2×2
        )

        # Bloque 5: 3 capas conv con 512 filtros
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2×2 -> 1×1
        )

        # Capas clasificadoras
        # Para CIFAR-10 (32×32), después de 5 poolings: 32 / 2^5 = 1
        # Por tanto: 512 × 1 × 1 = 512 características
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Inicialización de pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Inicializa pesos usando He initialization para capas con ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de VGG-16.

        Args:
            x: Tensor de entrada [B, 3, 32, 32]

        Returns:
            Logits de clasificación [B, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrae características antes de la capa de clasificación final.
        Útil para visualización de embeddings.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1)
        return x


print("Arquitectura VGG-16 definida correctamente")
```

En comparación con VGG-16 original, esta variante introduce varias adaptaciones
necesarias para CIFAR-10:

| Componente       | VGG-16 original           | Adaptación CIFAR-10     | Justificación                                      |
| ---------------- | ------------------------- | ----------------------- | -------------------------------------------------- |
| Tamaño entrada   | $224 \times 224 \times 3$ | $32 \times 32 \times 3$ | Tamaño propio de CIFAR-10                          |
| Capas FC         | 4096–4096–1000            | 512–512–10              | Menor resolución espacial y menor número de clases |
| BatchNorm        | No incluida originalmente | Incluida                | Estabiliza y acelera el entrenamiento              |
| Dropout          | 0.5 en FC                 | 0.5 en FC               | Regularización                                     |
| Número de clases | 1000 (ImageNet)           | 10 (CIFAR-10)           | Problema de clasificación diferente                |

La **inicialización de He (Kaiming)** se emplea en capas con activación ReLU. En términos
generales, establece los pesos con una distribución:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right),
$$

donde $\text{fan\_in}$ es el número de entradas a la capa. Esta estrategia reduce el
riesgo de desvanecimiento o explosión del gradiente en redes profundas al mantener la
varianza de las activaciones relativamente constante a través de las capas.

### Instanciación del modelo y análisis de complejidad

Definida la arquitectura, se instancia el modelo, se envía al dispositivo adecuado (CPU o
GPU) y se analiza su estructura y número de parámetros para verificar que la
implementación es correcta.

```python
# Crear el modelo
model = VGG16(num_classes=NUM_CLASSES)

# Determinar dispositivo disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Dispositivo utilizado: {device}")
print(f"\n{'=' * 70}")
print("RESUMEN DE LA ARQUITECTURA VGG-16")
print(f"{'=' * 70}\n")

# Resumen detallado de la arquitectura
summary(model, input_size=(BATCH_SIZE, 3, 32, 32), device=str(device))

# Conteo de parámetros por bloque
print(f"\n{'=' * 70}")
print("ANÁLISIS DE PARÁMETROS POR BLOQUE")
print(f"{'=' * 70}")


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


print(f"  Block 1: {count_parameters(model.block1):>12,} parámetros")
print(f"  Block 2: {count_parameters(model.block2):>12,} parámetros")
print(f"  Block 3: {count_parameters(model.block3):>12,} parámetros")
print(f"  Block 4: {count_parameters(model.block4):>12,} parámetros")
print(f"  Block 5: {count_parameters(model.block5):>12,} parámetros")
print(f"  Classifier: {count_parameters(model.classifier):>12,} parámetros")
print(f"  {'-' * 66}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  TOTAL: {total_params:>12,} parámetros")
print(f"  Entrenables: {trainable_params:>12,} parámetros")
print(f"  Memoria (float32): {total_params * 4 / (1024 ** 2):>10.2f} MB")
```

En la versión original de VGG-16 para ImageNet, alrededor de 14,7 millones de parámetros
corresponden a las capas convolucionales y unos 123 millones a las capas totalmente
conectadas (aproximadamente el 89 % del total). La variante para CIFAR-10 reduce
drásticamente los parámetros de las capas densas al pasar de 4096 a 512 neuronas, lo que
da lugar a un modelo más manejable y adecuado para este conjunto de datos, aunque aún
considerablemente grande en comparación con arquitecturas más modernas y eficientes.

### Configuración del entrenamiento

Se definen el optimizador, la función de pérdida y un planificador (_scheduler_) de la
tasa de aprendizaje para mejorar la convergencia y adaptarse dinámicamente a la evolución
del entrenamiento.

```python
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print(f"{'=' * 70}")
print(f"  Épocas: {NUM_EPOCHS}")
print(f"  Learning rate inicial: {LEARNING_RATE}")
print(f"  Weight decay (L2): {WEIGHT_DECAY}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"{'=' * 70}\n")

# Optimizador: SGD con momentum
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler: reduce LR cuando el progreso se estanca
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",     # Se monitoriza la accuracy (que se desea maximizar)
    factor=0.1,     # Reducir LR al 10 % del valor actual
    patience=3,     # Esperar 3 épocas sin mejora
    min_lr=1e-6
)

# Función de pérdida: Cross-Entropy
loss_function = nn.CrossEntropyLoss()

print("Optimizador: SGD con momentum=0.9")
print("  SGD con momentum acumula gradientes con decaimiento exponencial")
print("  Esto ayuda a superar mínimos locales y acelera la convergencia")
print("\nScheduler: ReduceLROnPlateau")
print("  Reduce el learning rate cuando la accuracy deja de mejorar")
print("  Factor de reducción: 0.1 (LR → 0.1 × LR)")
print("  Paciencia: 3 épocas")
print("\nFunción de pérdida: CrossEntropyLoss")
```

Históricamente, para arquitecturas como VGG, **SGD con momentum** se ha mostrado muy
eficaz, especialmente cuando se dispone de suficiente tiempo de entrenamiento y se ajusta
cuidadosamente la tasa de aprendizaje. El término de **momentum** acumula gradientes
pasados mediante:

$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla L(\theta_{t-1}), \quad
\theta_t = \theta_{t-1} - \text{lr} \cdot v_t,
$$

donde $0 < \beta < 1$ es el coeficiente de momentum (típicamente 0.9). Este mecanismo
acelera el descenso en direcciones de gradiente consistente y amortigua oscilaciones en
direcciones de alta curvatura, mejorando la velocidad de convergencia y la estabilidad
del entrenamiento.

El _scheduler_ `ReduceLROnPlateau` ajusta dinámicamente la tasa de aprendizaje en función
de la evolución del rendimiento en validación. Cuando la métrica monitorizada (en este
caso, la _accuracy_ de test) deja de mejorar durante un número determinado de épocas
(paciencia), el _scheduler_ reduce la tasa de aprendizaje, permitiendo un ajuste más fino
de los parámetros cerca de un óptimo local.

### Bucle de entrenamiento

El bucle de entrenamiento consta de dos fases por época: una fase de entrenamiento donde
se actualizan los parámetros del modelo, y una fase de validación donde se evalúa el
rendimiento sin modificar los pesos. Se registran las pérdidas y las precisiones en ambas
fases, así como la evolución de la tasa de aprendizaje.

```python
# Listas para almacenar métricas
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
learning_rates = []

# Función auxiliar para calcular accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total


print("INICIANDO ENTRENAMIENTO\n")
print(f"{'=' * 70}\n")

# Tiempo de inicio
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    # ============ FASE DE ENTRENAMIENTO ============
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

    # ============ FASE DE VALIDACIÓN ============
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

    # Actualizar learning rate scheduler
    scheduler.step(epoch_test_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

    # Calcular tiempo de época
    epoch_time = time.time() - epoch_start_time

    # Reporte de resultados
    print(f"Época [{epoch + 1}/{NUM_EPOCHS}] - Tiempo: {epoch_time:.2f}s")
    print(f"  Train → Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
    print(f"  Test  → Loss: {epoch_test_loss:.4f} | Acc: {epoch_test_acc:.2f}%")
    print(f"  LR: {current_lr:.6f}")
    print(f"  {'─' * 66}\n")

# Tiempo total
total_time = time.time() - start_time

print(f"\n{'=' * 70}")
print("ENTRENAMIENTO COMPLETADO")
print(f"{'=' * 70}")
print(f"  Tiempo total: {total_time / 60:.2f} minutos")
print(f"  Tiempo promedio por época: {total_time / NUM_EPOCHS:.2f} segundos")
print(f"  Precisión final (train): {train_accuracies[-1]:.2f}%")
print(f"  Precisión final (test): {test_accuracies[-1]:.2f}%")
print(f"  Mejor precisión (test): {max(test_accuracies):.2f}%")

# Guardar el modelo
torch.save({
    "epoch": NUM_EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "test_losses": test_losses,
    "test_accuracies": test_accuracies,
}, "vgg16_cifar10.pth")

print("\nModelo guardado como 'vgg16_cifar10.pth'")
```

VGG es computacionalmente intensivo debido al número de operaciones de convolución y al
elevado número de canales en las capas profundas. En una GPU moderna (por ejemplo, una
RTX 3080) cada época puede requerir del orden de decenas de segundos con la configuración
descrita, mientras que en CPU el proceso puede ser una orden de magnitud más lento. La
instrucción `model.train()` activa los comportamientos específicos de entrenamiento, como
la actualización de estadísticas en `BatchNorm` y la aplicación de `Dropout`, mientras
que `model.eval()` desactiva estos comportamientos para garantizar una evaluación
determinista y reproducible.

### Visualización de métricas de entrenamiento

La inspección de la evolución de la pérdida, la precisión y la tasa de aprendizaje a lo
largo de las épocas permite identificar posibles problemas de sobreajuste, estancamiento
del entrenamiento o configuraciones de _learning rate_ inadecuadas.

```python
epochs_range = range(1, NUM_EPOCHS + 1)

# Crear figura con tres subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Gráfica de pérdida
ax1.plot(epochs_range, train_losses, "o-", label="Train Loss",
         linewidth=2, markersize=6)
ax1.plot(epochs_range, test_losses, "s-", label="Test Loss",
         linewidth=2, markersize=6)
ax1.set_xlabel("Época", fontsize=12, fontweight="bold")
ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax1.set_title("Evolución de la Pérdida", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(list(epochs_range))

# Gráfica de precisión
ax2.plot(epochs_range, train_accuracies, "o-", label="Train Accuracy",
         linewidth=2, markersize=6)
ax2.plot(epochs_range, test_accuracies, "s-", label="Test Accuracy",
         linewidth=2, markersize=6)
ax2.set_xlabel("Época", fontsize=12, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax2.set_title("Evolución de la Precisión", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(list(epochs_range))

# Gráfica de learning rate
ax3.plot(epochs_range, learning_rates, "o-", color="red",
         linewidth=2, markersize=6)
ax3.set_xlabel("Época", fontsize=12, fontweight="bold")
ax3.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3)
ax3.set_xticks(list(epochs_range))

plt.tight_layout()
plt.savefig("vgg16_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

# Análisis cuantitativo
print("\nAnálisis de resultados:")
diff = train_accuracies[-1] - test_accuracies[-1]
print(f"  Overfitting detectado: {'SÍ' if diff > 10 else 'NO'}")
print(f"  Diferencia train-test: {diff:.2f}%")
print(f"  Mejor época (test acc): {np.argmax(test_accuracies) + 1}")
print(f"  Ganancia desde época 1: {test_accuracies[-1] - test_accuracies[0]:.2f}%")
```

Las curvas de pérdida y precisión permiten detectar comportamientos típicos: si la
pérdida de entrenamiento disminuye mientras la de validación aumenta, se evidencia
sobreajuste; si ambas permanecen altas, el modelo puede estar infraajustado o la tasa de
aprendizaje puede no ser adecuada. La gráfica de _learning rate_ en escala logarítmica
muestra los descensos escalonados producidos por el _scheduler_, que suelen coincidir con
fases de refinamiento del modelo donde se ajustan los parámetros de forma más precisa.

### Matriz de confusión y análisis por clase

La matriz de confusión proporciona una visión detallada del rendimiento por clase y
permite identificar patrones sistemáticos de error, facilitando la comprensión de qué
categorías resultan más difíciles de distinguir para el modelo.

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("Generando matriz de confusión...")

# Obtener todas las predicciones
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_dataloader, desc="Evaluando"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calcular matriz de confusión
cm = confusion_matrix(all_labels, all_predictions)

# Visualizar matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CIFAR10_CLASSES,
    yticklabels=CIFAR10_CLASSES,
    cbar_kws={"label": "Número de muestras"}
)
plt.xlabel("Predicción", fontsize=12, fontweight="bold")
plt.ylabel("Etiqueta Real", fontsize=12, fontweight="bold")
plt.title("Matriz de Confusión - VGG16 en CIFAR-10",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("vgg16_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print("=" * 70)
print(classification_report(
    all_labels, all_predictions,
    target_names=CIFAR10_CLASSES,
    digits=3
))

# Análisis de precisión por clase
print("\nAnálisis de Precisión por Clase:")
print("=" * 70)
class_correct = cm.diagonal()
class_total = cm.sum(axis=1)
class_accuracy = 100 * class_correct / class_total

for idx, class_name in enumerate(CIFAR10_CLASSES):
    print(
        f"  {class_name:12s}: {class_accuracy[idx]:6.2f}% "
        f"({class_correct[idx]:4d}/{class_total[idx]:4d})"
    )

# Identificar clases más confundidas
print("\nPares de Clases Más Confundidos:")
print("=" * 70)
confusion_pairs = []
for i in range(len(CIFAR10_CLASSES)):
    for j in range(len(CIFAR10_CLASSES)):
        if i != j:
            confusion_pairs.append((cm[i, j], CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]))

confusion_pairs.sort(reverse=True)
for count, true_class, pred_class in confusion_pairs[:5]:
    print(f"  {true_class:12s} → {pred_class:12s}: {count:4d} veces")
```

La diagonal principal de la matriz refleja las clasificaciones correctas, mientras que
los elementos fuera de la diagonal cuantifican las confusiones entre pares de clases. En
CIFAR-10 es habitual observar confusiones entre `cat` y `dog`, `automobile` y `truck`, o
`bird` y `airplane`, lo que evidencia que ciertas categorías comparten patrones visuales
similares desde la perspectiva del modelo. Este análisis resulta valioso para identificar
limitaciones del modelo y orientar posibles mejoras, como la recolección de datos
adicionales para clases problemáticas o la aplicación de técnicas de balanceo de clases.

### Visualización de predicciones correctas e incorrectas

Para entender con mayor detalle el comportamiento del modelo, resulta útil visualizar
algunas predicciones correctas y ejemplos de error, lo que proporciona una perspectiva
cualitativa complementaria a las métricas cuantitativas.

```python
print("Visualizando predicciones del modelo...\n")

# Obtener un batch de test
data_iter = iter(test_dataloader)
test_images, test_labels = next(data_iter)

# Hacer predicciones
model.eval()
with torch.no_grad():
    test_images_device = test_images.to(device)
    outputs = model(test_images_device)
    _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu()

# Visualizar primeras 8 predicciones
print("Primeras 8 predicciones:")
show_images(test_images[:8], test_labels[:8], predictions[:8])

# Encontrar ejemplos de error
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

Este análisis cualitativo complementa las métricas numéricas y facilita la detección de
patrones sistemáticos de error, como confundir sistemáticamente un tipo concreto de
vehículo con otro o ciertas clases de animales entre sí. La inspección visual de errores
puede revelar también problemas en los datos, como etiquetas incorrectas o imágenes
ambiguas que incluso para un humano resultarían difíciles de clasificar.

### Extracción y visualización de características intermedias

Una de las fortalezas de VGG es su capacidad para aprender características jerárquicas en
profundidad. Resulta posible inspeccionar las activaciones de capas intermedias para
comprender mejor qué tipo de patrones captura la red en cada bloque convolucional.

```python
def get_activation_maps(model, image, layer_name):
    """
    Extrae mapas de activación de una capa específica.
    """
    activations = {}

    def hook_fn(module, input, output):
        activations["output"] = output

    # Registrar hook
    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0).to(device))

    hook.remove()
    return activations["output"].squeeze().cpu()


# Seleccionar una imagen de prueba
test_image, test_label = test_dataset[0]
print(f"Analizando imagen de clase: {CIFAR10_CLASSES[test_label]}\n")

# Mostrar imagen original
plt.figure(figsize=(4, 4))
img_display = test_image / 2 + 0.5  # Desnormalizar
plt.imshow(img_display.permute(1, 2, 0))
plt.title(f"Imagen Original: {CIFAR10_CLASSES[test_label]}",
          fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.show()

# Seleccionar y visualizar activaciones de diferentes bloques
layers_to_visualize = {
    "block1": "block1.0",  # Primera conv de block1
    "block2": "block2.0",  # Primera conv de block2
    "block3": "block3.0",  # Primera conv de block3
    "block5": "block5.0",  # Primera conv de block5
}

for block_name, layer_name in layers_to_visualize.items():
    print(f"Visualizando activaciones de {block_name}...")

    activations = get_activation_maps(model, test_image, layer_name)

    # Seleccionar primeros 16 filtros para visualizar
    num_filters = min(16, activations.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(
        f"Activaciones de {block_name.upper()} - "
        f"{CIFAR10_CLASSES[test_label]}",
        fontsize=16, fontweight="bold"
    )

    for idx, ax in enumerate(axes.flat):
        if idx < num_filters:
            activation = activations[idx]
            ax.imshow(activation, cmap="viridis")
            ax.set_title(f"Filtro {idx}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"vgg16_activations_{block_name}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

print("\nInterpretación de activaciones por bloque:")
print("=" * 70)
print("  Block 1: Detecta características de bajo nivel")
print("           (bordes, esquinas, variaciones simples de color)")
print("  Block 2-3: Detecta patrones de nivel medio")
print("             (texturas, formas más complejas, patrones repetitivos)")
print("  Block 4-5: Detecta características de alto nivel")
print("             (partes de objetos, combinaciones de texturas y formas)")
```
