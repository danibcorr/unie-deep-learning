# LeNet

## Contexto histórico y relevancia de LeNet-5

La arquitectura **LeNet-5**, desarrollada por **Yann LeCun** y colaboradores entre 1988 y
1998, representa una de las primeras arquitecturas fundacionales en el ámbito de las
redes neuronales convolucionales. Su diseño se plantea para abordar de forma eficiente el
problema del reconocimiento automático de caracteres manuscritos. Esta arquitectura se
implementa con éxito en sistemas reales de procesamiento automático de cheques bancarios
en Estados Unidos, lo que la convierte en una de las primeras aplicaciones industriales
del aprendizaje profundo.

LeNet-5 demuestra que es posible aprender representaciones jerárquicas directamente a
partir de imágenes, preservando la estructura espacial de los datos y reduciendo
significativamente el número de parámetros con respecto a las redes densas tradicionales.
A diferencia de los perceptrones multicapa totalmente conectados utilizados previamente,
que tratan cada píxel como una característica independiente y pierden la información de
disposición espacial, LeNet-5 explota explícitamente la estructura bidimensional de las
imágenes y la correlación local entre píxeles próximos.

Esta arquitectura ilustra cómo la combinación de convoluciones, submuestreo y funciones
de activación no lineales permite construir sistemas robustos frente a variaciones en la
posición, escala o deformaciones moderadas del patrón manuscrito, manteniendo al mismo
tiempo una complejidad computacional manejable. De este modo, LeNet-5 se sitúa como
antecedente directo de muchas de las arquitecturas modernas de visión por computador,
estableciendo principios de diseño que persisten hasta la actualidad.

## Fundamentos conceptuales de LeNet-5

Antes de la introducción de LeNet-5, el reconocimiento de imágenes se aborda
principalmente mediante perceptrones multicapa totalmente conectados. Este enfoque
presenta una limitación estructural fundamental: la imagen se aplana en un vector
unidimensional, y el modelo pierde completamente la información acerca de la disposición
espacial relativa de los píxeles. Como consecuencia, el sistema se vuelve extremadamente
sensible a pequeñas traslaciones, deformaciones locales o cambios en la posición del
objeto dentro de la imagen. Además, el número de parámetros crece rápidamente con el
tamaño de la imagen, lo que dificulta el entrenamiento y favorece el sobreajuste,
especialmente cuando se dispone de conjuntos de datos limitados.

LeNet-5 introduce un cambio conceptual decisivo mediante el uso combinado de
convoluciones, submuestreo y compartición de pesos. Las capas convolucionales permiten
detectar patrones locales, como bordes, esquinas o trazos, preservando la estructura
bidimensional de la imagen. Cada filtro convolucional se desplaza sobre la imagen y actúa
como un detector especializado en cierto tipo de patrón visual. El submuestreo,
implementado mediante operaciones de _average pooling_, reduce progresivamente la
resolución espacial de los mapas de características. Esta reducción aporta una
invariancia aproximada frente a pequeñas traslaciones y deformaciones, y al mismo tiempo
disminuye el coste computacional y el número de parámetros de las capas posteriores.

La compartición de pesos constituye otro elemento clave del diseño: el mismo filtro
convolucional se aplica en todas las posiciones de la imagen. En lugar de aprender un
conjunto de pesos distinto para cada píxel de entrada, se aprende un conjunto reducido de
filtros que se reutilizan a lo largo de toda la imagen. Este mecanismo reduce de forma
drástica el número de parámetros y favorece la capacidad de generalización, ya que un
mismo patrón (por ejemplo, un trazo vertical) puede aparecer en distintas regiones de la
imagen y debe ser reconocido de manera coherente en todas ellas.

En conjunto, estos mecanismos permiten que LeNet-5 aprenda representaciones jerárquicas
de manera progresiva. Las capas iniciales capturan patrones simples y locales, mientras
que las capas más profundas combinan dichos patrones para formar características de mayor
nivel, cada vez más abstractas y específicas de la tarea de clasificación. Esta
organización jerárquica de características se mantiene como principio básico en la
mayoría de arquitecturas modernas de redes neuronales convolucionales, desde AlexNet
hasta las arquitecturas de transformadores aplicados a visión.

## Arquitectura original de LeNet-5

La arquitectura original de LeNet-5 se compone de siete capas entrenables que combinan
convoluciones, submuestreo y capas totalmente conectadas. La entrada está formada por
imágenes en escala de grises de tamaño $32 \times 32$, ligeramente mayor que el tamaño
estándar de las imágenes de MNIST, que es de $28 \times 28$. Esta ampliación introduce un
margen alrededor del dígito, lo que facilita la aplicación de filtros convolucionales y
el tratamiento de pequeñas traslaciones sin pérdida de información relevante en los
bordes de la imagen.

De manera simplificada, la arquitectura se organiza en una parte convolucional seguida de
una parte totalmente conectada. La parte convolucional alterna capas de convolución y
_average pooling_, mientras que la parte final consta de capas densas que realizan la
clasificación. Las dimensiones características de cada capa pueden resumirse del
siguiente modo:

| Capa   | Tipo            | Entrada                  | Salida                   |
| ------ | --------------- | ------------------------ | ------------------------ |
| C1     | Convolución     | $32 \times 32 \times 1$  | $28 \times 28 \times 6$  |
| S2     | Average Pooling | $28 \times 28 \times 6$  | $14 \times 14 \times 6$  |
| C3     | Convolución     | $14 \times 14 \times 6$  | $10 \times 10 \times 16$ |
| S4     | Average Pooling | $10 \times 10 \times 16$ | $5 \times 5 \times 16$   |
| C5     | Convolución     | $5 \times 5 \times 16$   | $1 \times 1 \times 120$  |
| F6     | Fully Connected | 120                      | 84                       |
| Output | Fully Connected | 84                       | 10                       |

La arquitectura original utiliza funciones de activación sigmoide o $\tanh$ y aplica
_average pooling_ en lugar de _max pooling_, técnica que se populariza posteriormente en
arquitecturas más modernas. El número total de parámetros entrenables se sitúa en torno a
los 60 000, una cifra relativamente contenida si se compara con redes totalmente
conectadas de tamaño similar. Pese a su aparente sencillez, LeNet-5 establece una pauta
estructural que se mantiene en muchas arquitecturas contemporáneas: una parte
convolucional encargada de la extracción de características espaciales, seguida de una o
varias capas densas que realizan la clasificación final mediante combinaciones lineales
de las características extraídas.

## Implementación en PyTorch

En esta sección se presenta una implementación moderna y funcional inspirada en LeNet,
utilizando PyTorch. El objetivo es disponer de un flujo de trabajo completo, ejecutable
de principio a fin y directamente convertible en un Jupyter Notebook. Se emplea el
dataset MNIST como conjunto de datos de referencia para la clasificación de dígitos
manuscritos.

### Importación de bibliotecas

En primer lugar, se importan las bibliotecas necesarias, tanto de la biblioteca estándar
de Python como de terceros. Se incluyen módulos para la construcción del modelo, el
manejo de datos, la visualización y el análisis de embeddings.

```python
# Bibliotecas estándar
from typing import Any

# Bibliotecas de terceros
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.manifold import TSNE

print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

### Configuración del dispositivo

A continuación se selecciona automáticamente el dispositivo en función de la
disponibilidad de GPU. Si CUDA está disponible, se emplea la GPU; en caso contrario, el
modelo se ejecuta en CPU.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo utilizado: {device}")
```

### Función auxiliar de visualización

Para facilitar la inspección visual de ejemplos del dataset, se define una función que
muestra un conjunto de imágenes junto con sus etiquetas correspondientes. Esta
visualización permite comprobar de forma rápida que el preprocesamiento es correcto y que
las muestras se interpretan adecuadamente.

```python
def show_images(images, labels):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    if len(images) == 1:
        axes = [axes]

    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Dígito: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
```

## Carga y preprocesamiento del dataset MNIST

Se utiliza el dataset MNIST, que contiene imágenes de dígitos manuscritos en escala de
grises de tamaño $28 \times 28$. Como parte del preprocesamiento se aplica una
normalización utilizando la media $\mu = 0.1307$ y la desviación estándar
$\sigma = 0.3081$, valores estimados sobre el propio conjunto de datos. La normalización
se define como:

$$
x_{\text{normalizado}} = \frac{x - \mu}{\sigma}.
$$

Esta operación centra los datos y ajusta su escala, lo que facilita y estabiliza el
proceso de entrenamiento de redes profundas al mejorar el condicionamiento numérico de
las operaciones de optimización.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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

print(f"Muestras de entrenamiento: {len(train_dataset)}")
print(f"Muestras de prueba: {len(test_dataset)}")
```

### Creación de DataLoaders

A partir de los conjuntos de entrenamiento y prueba se construyen objetos `DataLoader`.
Estos se encargan de agrupar las muestras en _batches_, barajar los ejemplos en
entrenamiento y gestionar la transferencia de datos al dispositivo de cómputo de manera
eficiente.

```python
BATCH_SIZE = 32

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)
```

### Inspección visual del dataset

Antes de proceder al entrenamiento, resulta conveniente inspeccionar algunas muestras del
conjunto de entrenamiento. Además de visualizar las imágenes, se calcula la media y la
desviación estándar de un _batch_ para verificar que la normalización se aplica
correctamente.

```python
images, labels = next(iter(train_dataloader))
show_images(images[:10], labels[:10])

print(f"Media del batch: {images.mean():.3f}")
print(f"Desviación estándar del batch: {images.std():.3f}")
```

## Definición de una versión moderna de LeNet en PyTorch

A continuación se define una versión moderna y simplificada de LeNet, adaptada a MNIST y
a las prácticas actuales en aprendizaje profundo. Aunque no reproduce exactamente la
arquitectura original de LeNet-5, mantiene el espíritu del diseño: una parte
convolucional que extrae características espaciales y una parte totalmente conectada que
realiza la clasificación. Se incorporan capas de normalización por lotes (_Batch
Normalization_) y la función de activación ReLU, que son estándar en arquitecturas
contemporáneas y han demostrado mejorar tanto la velocidad de convergencia como la
estabilidad del entrenamiento.

```python
class LeNet(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
```

En esta definición, el bloque `features` aplica dos capas convolucionales con reducción
de resolución espacial mediante `stride=2`, seguidas de normalización por lotes y
activación ReLU. Posteriormente, la capa `nn.AdaptiveAvgPool2d((1, 1))` reduce de manera
adaptativa cada mapa de características a tamaño $1 \times 1$ por canal. Este mecanismo
hace que la arquitectura resulte robusta a pequeños cambios en el tamaño espacial de la
entrada, siempre que se mantenga una estructura razonablemente similar.

El bloque `classifier` aplana las características resultantes y aplica una capa lineal
que produce las puntuaciones (logits) para las 10 clases de dígitos. Estas puntuaciones
se interpretan posteriormente mediante `CrossEntropyLoss`, que incorpora internamente la
operación de _softmax_ para calcular la distribución de probabilidades sobre las clases.

### Instanciación y análisis del modelo

Se instancia el modelo, se traslada al dispositivo de cómputo seleccionado y se utiliza
`torchinfo.summary` para obtener un resumen estructurado de la arquitectura. Este resumen
incluye el tamaño de las entradas y salidas de cada capa y el número de parámetros
asociados, lo que permite verificar que las dimensiones son coherentes y que la
complejidad del modelo es razonable.

```python
model = LeNet().to(device)

summary(
    model,
    input_size=(BATCH_SIZE, 1, 28, 28),
    device=str(device)
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total de parámetros entrenables: {total_params:,}")
```

## Configuración del proceso de entrenamiento

Una vez definida la arquitectura, se especifican los hiperparámetros de entrenamiento, el
optimizador y la función de pérdida. En este caso se utiliza el optimizador AdamW, que
combina las ventajas de Adam con una regularización explícita mediante _weight decay_. La
función de pérdida elegida es `CrossEntropyLoss`, adecuada para problemas de
clasificación multiclase con etiquetas enteras.

```python
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

loss_function = nn.CrossEntropyLoss()
```

## Bucle de entrenamiento y validación

El proceso de entrenamiento se organiza en épocas. En cada época se recorre el conjunto
de entrenamiento para actualizar los parámetros del modelo, y posteriormente se evalúa el
rendimiento en el conjunto de prueba sin modificar dichos parámetros. Se registran tanto
la pérdida como la precisión en entrenamiento y en prueba, lo que permite analizar la
evolución del aprendizaje y detectar posibles problemas como el sobreajuste.

```python
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_dataloader, desc=f"Época {epoch+1}/{NUM_EPOCHS} [TRAIN]"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_losses.append(running_loss / len(train_dataloader))
    train_accuracies.append(100 * correct / total)

    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc=f"Época {epoch+1}/{NUM_EPOCHS} [TEST]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    test_losses.append(test_loss / len(test_dataloader))
    test_accuracies.append(100 * correct / total)

    print(f"Época {epoch+1}")
    print(f"  Train → Loss: {train_losses[-1]:.4f} | Acc: {train_accuracies[-1]:.2f}%")
    print(f"  Test  → Loss: {test_losses[-1]:.4f} | Acc: {test_accuracies[-1]:.2f}%")
```

En este bucle, la instrucción `model.train()` activa los comportamientos específicos de
entrenamiento, como la actualización interna de las estadísticas en `BatchNorm` y la
aplicación de _dropout_ si estuviera presente. Por el contrario, `model.eval()` desactiva
dichos comportamientos para garantizar una evaluación coherente y determinista. El
contexto `torch.no_grad()` durante la validación evita el cálculo de gradientes, lo que
reduce significativamente el consumo de memoria y el tiempo de cómputo.

## Visualización de la evolución de las métricas

Tras el entrenamiento, se representa gráficamente la evolución de las pérdidas y las
precisiones en entrenamiento y en prueba. Este análisis visual permite identificar
fenómenos como el sobreajuste, el subajuste o el estancamiento del aprendizaje,
facilitando la toma de decisiones sobre posibles ajustes en la arquitectura o en los
hiperparámetros.

```python
epochs = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("Evolución de la pérdida")

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Evolución de la precisión")

plt.tight_layout()
plt.show()
```

La comparación entre las curvas de entrenamiento y prueba proporciona información valiosa
sobre la capacidad de generalización del modelo. Por ejemplo, un incremento continuado de
la precisión en entrenamiento acompañado de una precisión de prueba estancada o
decreciente suele indicar sobreajuste, mientras que pérdidas altas en ambos conjuntos
sugieren que la capacidad del modelo o el tiempo de entrenamiento son insuficientes para
resolver adecuadamente el problema planteado.

## Visualización de embeddings mediante t-SNE

Finalmente, se analiza la estructura de los embeddings producidos por el modelo
utilizando la técnica de reducción de dimensionalidad t-SNE (_t-distributed Stochastic
Neighbor Embedding_). En este caso, se extraen directamente las salidas lineales del
modelo (logits) como representaciones de los ejemplos. La técnica t-SNE proyecta estos
vectores de alta dimensión en un espacio bidimensional, preservando en lo posible las
relaciones de vecindad locales. Esta proyección facilita la inspección visual de cómo el
modelo separa las distintas clases en el espacio de características.

```python
model.eval()

max_samples = 1000
embeddings, all_labels = [], []

with torch.no_grad():
    for i, (images, labels) in enumerate(train_dataloader):
        if len(all_labels) * train_dataloader.batch_size >= max_samples:
            break
        images = images.to(device)
        outputs = model(images)
        embeddings.append(outputs.cpu())
        all_labels.append(labels)

embeddings = torch.cat(embeddings).numpy()
all_labels = torch.cat(all_labels).numpy()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=300,
    learning_rate=200,
    n_jobs=-1
)
X_embedded = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_embedded[:, 0],
    X_embedded[:, 1],
    c=all_labels,
    cmap="tab10",
    alpha=0.6,
    s=10
)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE de embeddings aprendidos por LeNet")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.show()
```

Cuando el modelo ha aprendido una representación adecuada de los datos, se observa que
los puntos correspondientes a distintas clases tienden a agruparse formando cúmulos
relativamente separados en el espacio bidimensional. Esta visualización proporciona una
perspectiva intuitiva sobre cómo el modelo organiza internamente la información y cómo
distingue entre las diferentes clases de dígitos manuscritos. En particular, la
separación clara entre grupos indica que el espacio de características inducido por la
red facilita la clasificación lineal en la capa final, confirmando que las
representaciones aprendidas son discriminativas y semánticamente coherentes.
