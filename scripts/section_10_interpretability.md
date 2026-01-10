# Interpretabilidad en redes neuronales convolucionales

La interpretabilidad en redes neuronales convolucionales (CNN) resulta esencial para
comprender las decisiones del modelo, detectar posibles sesgos y mejorar su robustez
frente a perturbaciones o cambios de distribución. Debido a su profundidad y a la
combinación de convoluciones y funciones de activación no lineales, las CNN se comportan
como sistemas de alta complejidad, cuya inspección directa resulta limitada. Por este
motivo se desarrollan técnicas específicas que permiten visualizar qué regiones o
características de la señal de entrada contribuyen de manera más relevante a las
predicciones.

En este texto se describen e implementan varias de las metodologías más utilizadas para
interpretar CNN: mapas de saliencia (_saliency maps_), Grad-CAM, Guided Grad-CAM,
análisis por oclusión e Integrated Gradients. Posteriormente se presenta una
implementación funcional completa sobre CIFAR-10 utilizando un modelo ResNet-18 adaptado,
organizada de forma lineal para su ejecución paso a paso, fácilmente convertible en un
cuaderno Jupyter.

## Mapas de saliencia (Saliency Maps)

Los mapas de saliencia se basan en el cálculo del gradiente de la salida del modelo con
respecto a cada píxel de la imagen de entrada. De forma intuitiva, si una pequeña
variación en un píxel produce un cambio significativo en la salida asociada a una clase
concreta, dicho píxel se considera importante para la decisión. El valor absoluto de ese
gradiente se utiliza como medida de relevancia local.

Dado un modelo $f(\cdot)$ y una imagen de entrada $x$, el mapa de saliencia para una
clase $c$ se define como

$$
S = \left| \frac{\partial f_c(x)}{\partial x} \right|.
$$

Cuando la red opera sobre entradas con varios canales (por ejemplo, imágenes RGB),
resulta habitual agregar la información de canal para construir un mapa bidimensional.
Una forma sencilla de hacerlo es tomar el máximo sobre el eje de canales:

$$
S_{i,j} = \max_{k} \left| \frac{\partial f_c(x)}{\partial x_{k,i,j}} \right|.
$$

Este mapa proporciona, para cada posición espacial $(i,j)$, una medida de sensibilidad de
la puntuación de clase $f_c$ frente a perturbaciones de los píxeles correspondientes. Los
mapas de saliencia se caracterizan por su sencillez conceptual y su eficiencia
computacional, aunque sus visualizaciones pueden resultar ruidosas y no siempre se
alinean claramente con regiones semánticamente interpretables de la imagen.

A continuación se presenta una clase que genera mapas de saliencia mediante gradientes y
una función de visualización que superpone el mapa resultante sobre la imagen original.

```python
"""
Interpretability in Convolutional Neural Networks
Implementación funcional completa con CIFAR-10

Técnicas implementadas:
1. Saliency Maps
2. Grad-CAM
3. Guided Grad-CAM (a partir de Guided Backpropagation)
4. Occlusion Analysis
5. Integrated Gradients
"""

# IMPORTACIONES
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}\n")

# CONFIGURACIÓN GLOBAL
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
}

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"Configuración: {CONFIG}\n")
```

## Preparación de datos CIFAR-10

Para ilustrar las distintas técnicas de interpretabilidad se utiliza el conjunto de datos
CIFAR-10, que contiene imágenes a color de tamaño $32 \times 32$ de diez clases
distintas. El siguiente código descarga y prepara el conjunto de prueba, aplicando una
normalización estándar ampliamente utilizada para este dataset.

```python
def prepare_cifar10_data():
    """
    Descarga y prepara el conjunto de prueba de CIFAR-10
    con normalización estándar.
    """
    print("Preparando CIFAR-10...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2
    )

    print(f"Test: {len(test_dataset)} imágenes\n")
    return test_loader, test_dataset
```

## Modelo ResNet-18 adaptado a CIFAR-10

Se emplea como base una ResNet-18 preentrenada en ImageNet, adaptada a las
características de CIFAR-10. La adaptación consiste en modificar la primera capa
convolucional para trabajar de forma más adecuada con imágenes de tamaño $32 \times 32$ y
ajustar la capa completamente conectada final al número de clases del conjunto CIFAR-10.
Aunque el modelo se carga con pesos preentrenados en ImageNet, la capa final se
inicializa de manera aleatoria, de modo que el rendimiento puede no ser óptimo si no se
realiza un ajuste fino (_fine-tuning_). No obstante, este detalle no afecta al propósito
principal de este código, que es ilustrar las técnicas de interpretabilidad de forma
funcional.

```python
def load_pretrained_model():
    """
    Carga una ResNet-18 preentrenada en ImageNet y la adapta a CIFAR-10.
    """
    print("Cargando modelo preentrenado...")

    model = models.resnet18(pretrained=True)

    # Adaptar primera capa a imágenes 32x32 (sin max-pooling inicial)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()

    # Adaptar capa final para 10 clases de CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(CONFIG["device"])
    model.eval()

    print(f"Modelo cargado en {CONFIG['device']}\n")
    return model
```

## Mapas de saliencia: implementación y visualización

Se presenta a continuación una implementación de mapas de saliencia basada en gradientes,
junto con una rutina de visualización que permite analizar de forma directa qué regiones
de la imagen contribuyen más a la predicción del modelo.

```python
print("=" * 70)
print("1. SALIENCY MAPS")
print("=" * 70)
print("""
Los mapas de saliencia calculan el gradiente de la salida respecto
a cada píxel de la imagen de entrada, indicando qué regiones tienen
mayor influencia en la predicción.

Ventajas:
- Cálculo sencillo y eficiente.
- Muestra influencia directa de los píxeles.

Limitaciones:
- Visualizaciones a menudo ruidosas.
- No siempre se alinean con regiones semánticamente claras.
""")


class SaliencyMapGenerator:
    """Genera mapas de saliencia mediante gradientes."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def generate_saliency(self, image: torch.Tensor, target_class: int | None = None):
        """
        Calcula el mapa de saliencia para una imagen.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            target_class: Índice de clase objetivo; si es None, se usa la predicción del modelo.

        Returns:
            Mapa de saliencia 2D (numpy array).
        """
        image = image.to(self.device)
        image.requires_grad = True

        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, target_class].backward()

        saliency = image.grad.data.abs()
        # Agregación por canales: máximo a lo largo del eje de canales
        saliency, _ = torch.max(saliency, dim=1)

        return saliency.squeeze().cpu().numpy()

    def visualize_saliency(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: int | None = None
    ) -> None:
        """
        Visualiza el mapa de saliencia y su superposición sobre la imagen original.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            original_image: Imagen desnormalizada [H, W, 3] en [0, 1].
            target_class: Clase objetivo; si es None, se utiliza la predicción del modelo.
        """
        saliency = self.generate_saliency(image, target_class)

        # Normalizar a [0, 1] para visualización
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(saliency, cmap="hot")
        axes[1].set_title("Saliency Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(saliency, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
```

## Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM genera mapas de calor que localizan las regiones de una imagen que contribuyen
de manera más destacada a la predicción de una clase específica. En lugar de operar
directamente sobre los píxeles de entrada, Grad-CAM trabaja sobre los mapas de activación
de una capa convolucional interna, lo que tiende a producir mapas de relevancia
espacialmente más estructurados y con mayor interpretabilidad semántica.

Sea $A^k \in \mathbb{R}^{H \times W}$ el mapa de activación asociado al canal $k$ de una
determinada capa convolucional. Para una clase $c$, se calculan coeficientes de
importancia mediante la media global de los gradientes sobre la dimensión espacial:

$$
\alpha_k = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \frac{\partial f_c}{\partial A^k_{ij}}.
$$

A partir de estos coeficientes se construye un mapa de activación ponderada para la clase
$c$:

$$
L_c^{\text{Grad-CAM}} = \mathrm{ReLU}\left( \sum_k \alpha_k A^k \right).
$$

La función ReLU se aplica para mantener únicamente las contribuciones positivas, bajo la
hipótesis de que aquellas activaciones que aumentan la puntuación de clase son las que se
desean destacar. La resolución espacial del mapa Grad-CAM está limitada por el tamaño de
los mapas de activación de la capa seleccionada; por ello, a menudo se interpola el
resultado para ajustarlo al tamaño de la imagen original.

La implementación siguiente utiliza _hooks_ para registrar activaciones y gradientes en
la capa objetivo y generar el mapa Grad-CAM correspondiente.

```python
print("=" * 70)
print("2. GRAD-CAM (Gradient-weighted Class Activation Mapping)")
print("=" * 70)
print("""
Grad-CAM genera mapas de calor que muestran las regiones de la imagen
más importantes para una clase específica, utilizando gradientes hacia
una capa convolucional interna.

Ventajas:
- Mapas más interpretables que los mapas de saliencia básicos.
- Localiza regiones relevantes del objeto de interés.

Limitaciones:
- Depende de la elección de la capa objetivo.
- Resolución limitada por la resolución de dicha capa.
""")


class GradCAM:
    """Implementación de Grad-CAM para una capa objetivo de una CNN."""

    def __init__(self, model: nn.Module, target_layer: str, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.target_layer = target_layer
        self.device = device

        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Registra hooks en la capa objetivo para capturar activaciones y gradientes
        durante la propagación hacia adelante y hacia atrás.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate_cam(self, image: torch.Tensor, target_class: int | None = None):
        """
        Genera el mapa Grad-CAM para una imagen y una clase objetivo.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            target_class: Clase objetivo; si es None, se utiliza la predicción del modelo.

        Returns:
            cam: Mapa Grad-CAM 2D normalizado (numpy array).
            target_class: Clase utilizada para la explicación.
        """
        self.model.eval()
        image = image.to(self.device)

        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Cálculo de pesos: media global de los gradientes sobre H x W
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class

    def visualize_cam(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: int | None = None
    ):
        """
        Visualiza el mapa Grad-CAM y su superposición sobre la imagen original.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            original_image: Imagen desnormalizada [H, W, 3] en [0, 1].
            target_class: Clase objetivo; si es None, se utiliza la predicción.
        """
        cam, pred_class = self.generate_cam(image, target_class)

        # Ajustar tamaño del mapa a la imagen original mediante interpolación
        cam_resized = zoom(
            cam,
            (
                original_image.shape[0] / cam.shape[0],
                original_image.shape[1] / cam.shape[1]
            )
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title(
            f"Grad-CAM (Class: {CIFAR10_CLASSES[pred_class]})",
            fontsize=12, fontweight="bold"
        )
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

        return cam_resized, pred_class
```

## Guided Backpropagation y Guided Grad-CAM

Guided Backpropagation modifica el flujo de gradientes a través de las unidades ReLU,
forzando a cero aquellos gradientes que son negativos tanto en la activación como en el
gradiente entrante. Este filtrado produce mapas de gradiente más nítidos y centrados en
características consideradas relevantes.

Guided Grad-CAM combina la capacidad de localización global de Grad-CAM con el detalle de
píxel de Guided Backpropagation. El procedimiento habitual se estructura en tres pasos
encadenados: primero se calcula un mapa Grad-CAM para la clase objetivo; después se
obtienen los gradientes guiados respecto a la imagen de entrada; finalmente se interpola
el mapa Grad-CAM hasta la resolución de la imagen y se realiza un producto elemento a
elemento con los gradientes guiados. El resultado es una visualización de alta resolución
en la que se enfatizan bordes y detalles finos dentro de las regiones que Grad-CAM
identifica como relevantes.

El código que sigue implementa Guided Backpropagation. Esta implementación se integra
fácilmente con la clase `GradCAM` anterior para componer Guided Grad-CAM, si se desea,
multiplicando el mapa Grad-CAM redimensionado por los gradientes guiados.

```python
print("=" * 70)
print("3. GUIDED GRAD-CAM")
print("=" * 70)
print("""
Guided Grad-CAM combina Grad-CAM con Guided Backpropagation
para obtener visualizaciones de alta resolución que son
espacialmente precisas y detalladas a nivel de píxel.

En este script se implementa Guided Backpropagation,
que puede combinarse con mapas Grad-CAM.
""")


class GuidedBackprop:
    """Implementación de Guided Backpropagation sobre una CNN."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.device = device
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Registra hooks en capas ReLU para filtrar gradientes negativos
        durante la propagación hacia atrás.
        """
        def backward_hook(module, grad_input, grad_output):
            if len(grad_input) > 0 and grad_input[0] is not None:
                return (torch.clamp(grad_input[0], min=0.0),)
            return grad_input

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_full_backward_hook(backward_hook)

    def generate_gradients(self, image: torch.Tensor, target_class: int | None = None):
        """
        Genera gradientes guiados respecto a la imagen de entrada.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            target_class: Clase objetivo; si es None, se usa la predicción.

        Returns:
            Gradientes guiados como numpy array [3, H, W].
        """
        self.model.eval()
        image = image.to(self.device)
        image.requires_grad = True

        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = image.grad.data.cpu().numpy()[0]
        return gradients
```

## Análisis por oclusión (Occlusion Analysis)

El análisis por oclusión adopta una perspectiva complementaria a los métodos basados en
gradientes: en lugar de estudiar la sensibilidad interna del modelo, modifica
explícitamente la entrada. En particular, se ocultan de forma sistemática pequeñas
regiones (parches) de la imagen y se mide el efecto sobre la probabilidad asignada a una
clase concreta. Cuando la oclusión de una región produce una disminución notable en la
probabilidad, se interpreta que dicha región resulta importante para la predicción.

Formalmente, para cada posición $(i,j)$ de una ventana deslizante se construye una
versión ocluida de la imagen $x^{(i,j)}$ y se evalúa la diferencia

$$
\Delta p_c^{(i,j)} = p_c(x) - p_c\bigl(x^{(i,j)}\bigr),
$$

donde $p_c(x)$ denota la probabilidad asignada a la clase $c$ por el modelo. El mapa de
sensibilidad resultante cuantifica, de forma directa, la importancia de cada región de la
imagen en términos de su impacto sobre la confianza del modelo. Esta técnica es
independiente de los gradientes y de detalles arquitectónicos específicos, aunque su
coste computacional se incrementa a medida que crece la resolución de la imagen, debido
al elevado número de evaluaciones requeridas.

La siguiente clase implementa un análisis por oclusión sencillo, permitiendo ajustar el
tamaño del parche de oclusión y el _stride_ de la ventana deslizante.

```python
print("=" * 70)
print("4. OCCLUSION ANALYSIS")
print("=" * 70)
print("""
Oculta sistemáticamente regiones de la imagen para observar
cómo cambia la predicción, revelando qué áreas son críticas.

Ventajas:
- Interpretación directa sobre la entrada del modelo.
- No requiere gradientes ni acceso interno a la arquitectura.

Limitaciones:
- Coste computacional elevado.
- Sensible al tamaño del parche y al stride utilizado.
""")


class OcclusionAnalysis:
    """Análisis por oclusión para la obtención de mapas de sensibilidad."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def analyze(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        patch_size: int = 4,
        stride: int = 2
    ):
        """
        Calcula un mapa de sensibilidad mediante oclusión sistemática.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            target_class: Clase objetivo; si es None, se utiliza la predicción.
            patch_size: Tamaño del parche cuadrado de oclusión en píxeles.
            stride: Desplazamiento de la ventana de oclusión.

        Returns:
            Mapa 2D de sensibilidad (numpy array).
        """
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            baseline_prob = torch.softmax(output, dim=1)[0, target_class].item()

        _, _, h, w = image.shape
        sensitivity_map = np.zeros((h, w))

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                occluded_image = image.clone()
                occluded_image[:, :, i:i + patch_size, j:j + patch_size] = 0

                with torch.no_grad():
                    output = self.model(occluded_image)
                    prob = torch.softmax(output, dim=1)[0, target_class].item()

                sensitivity = baseline_prob - prob
                current = sensitivity_map[i:i + patch_size, j:j + patch_size].mean()
                sensitivity_map[i:i + patch_size, j:j + patch_size] = max(current, sensitivity)

        return sensitivity_map

    def visualize(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: int | None = None,
        patch_size: int = 4,
        stride: int = 2
    ) -> None:
        """
        Visualiza el mapa de sensibilidad obtenido mediante oclusión.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            original_image: Imagen desnormalizada [H, W, 3] en [0, 1].
            target_class: Clase objetivo; si es None, se utiliza la predicción.
            patch_size: Tamaño del parche de oclusión.
            stride: Desplazamiento de la ventana de oclusión.
        """
        print(f"Analizando con patch_size={patch_size}, stride={stride}...")
        sensitivity = self.analyze(image, target_class, patch_size, stride)

        sensitivity = (sensitivity - sensitivity.min()) / \
            (sensitivity.max() - sensitivity.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(sensitivity, cmap="hot")
        axes[1].set_title("Sensitivity Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(sensitivity, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
```

## Integrated Gradients

Integrated Gradients es un método con fundamento teórico sólido para atribuir la
predicción de un modelo a las características de entrada. En lugar de considerar
únicamente el gradiente en el punto $x$, este método integra los gradientes a lo largo de
una trayectoria continua que conecta una _baseline_ $x'$ (por ejemplo, una imagen
completamente negra) con la imagen real $x$. De esta forma se mitigan problemas de
saturación de gradientes y se satisfacen axiomas de atribución deseables como la
sensibilidad y la invariancia a la implementación.

Sea $f_c$ la puntuación de la clase $c$ (por ejemplo, la salida antes de la capa
softmax). Integrated Gradients para la dimensión $i$ se define como

$$
\mathrm{IG}_i(x) = (x_i - x'_i) \int_{\alpha=0}^{1}
\frac{\partial f_c\bigl(x' + \alpha (x - x')\bigr)}{\partial x_i} \, d\alpha.
$$

En la práctica, la integral se aproxima mediante una suma discreta sobre $m$ pasos
uniformes:

$$
\mathrm{IG}_i(x) \approx (x_i - x'_i) \cdot \frac{1}{m} \sum_{k=1}^{m}
\frac{\partial f_c\bigl(x' + \tfrac{k}{m}(x - x')\bigr)}{\partial x_i}.
$$

La agregación de las contribuciones absolutas a lo largo de los canales produce un mapa
de relevancia espacial que suele ser más estable que el de los mapas de saliencia
básicos, a costa de requerir múltiples evaluaciones del modelo a lo largo de la
trayectoria definida entre la _baseline_ y la imagen original.

La implementación siguiente calcula Integrated Gradients para una imagen, permitiendo
especificar la _baseline_ y el número de pasos de integración.

```python
print("=" * 70)
print("5. INTEGRATED GRADIENTS")
print("=" * 70)
print("""
Método que atribuye la predicción a las características de entrada
integrando gradientes a lo largo de un camino desde una baseline
hasta la imagen real.

Ventajas:
- Fundamento teórico sólido.
- Reduce problemas de saturación de gradientes.

Limitaciones:
- Requiere múltiples evaluaciones del modelo.
- Depende de la elección de la baseline.
""")


class IntegratedGradients:
    """Implementación de Integrated Gradients para modelos en PyTorch."""

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def generate(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        baseline: torch.Tensor | None = None,
        steps: int = 50
    ):
        """
        Calcula Integrated Gradients para una imagen y una clase objetivo.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            target_class: Clase objetivo; si es None, se toma la predicción.
            baseline: Tensor [1, 3, H, W] que actúa como referencia. Si es None, se usa un tensor de ceros.
            steps: Número de puntos de muestreo en la trayectoria de integración.

        Returns:
            Numpy array [C, H, W] con las atribuciones por canal.
        """
        if baseline is None:
            baseline = torch.zeros_like(image)

        baseline = baseline.to(self.device)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Trayectoria lineal entre baseline e imagen
        scaled_inputs = [
            baseline + (float(i) / steps) * (image - baseline)
            for i in range(steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad = True

        output = self.model(scaled_inputs)
        self.model.zero_grad()
        target_output = output[:, target_class]
        target_output.backward(torch.ones_like(target_output))

        gradients = scaled_inputs.grad
        avg_gradients = torch.mean(gradients, dim=0, keepdim=True)

        integrated_grads = (image - baseline) * avg_gradients

        return integrated_grads.squeeze().cpu().detach().numpy()

    def visualize(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: int | None = None
    ) -> None:
        """
        Visualiza Integrated Gradients agregados espacialmente y su superposición.

        Args:
            image: Tensor [1, 3, H, W] normalizado.
            original_image: Imagen desnormalizada [H, W, 3] en [0, 1].
            target_class: Clase objetivo; si es None, se utiliza la predicción del modelo.
        """
        print("Calculando Integrated Gradients (50 steps)...")
        ig = self.generate(image, target_class)

        ig_aggregated = np.sum(np.abs(ig), axis=0)
        ig_aggregated = (ig_aggregated - ig_aggregated.min()) / \
            (ig_aggregated.max() - ig_aggregated.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(ig_aggregated, cmap="hot")
        axes[1].set_title("Integrated Gradients", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(original_image)
        axes[2].imshow(ig_aggregated, cmap="hot", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
```

## Utilidades de visualización

Para interpretar correctamente los resultados, es conveniente desnormalizar las imágenes
de CIFAR-10 antes de mostrarlas. La siguiente función revierte la normalización estándar
aplicada en la etapa de preprocesamiento y devuelve una imagen en formato adecuado para
`matplotlib`.

```python
def denormalize_cifar10(tensor: torch.Tensor) -> np.ndarray:
    """
    Desnormaliza un tensor de CIFAR-10 para su visualización.

    Args:
        tensor: Tensor [3, H, W] normalizado con la media y desviación estándar de CIFAR-10.

    Returns:
        Imagen en formato numpy [H, W, 3] con valores en [0, 1].
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0, 1)

    return denorm.permute(1, 2, 0).numpy()
```

## Pipeline completo de interpretabilidad

Finalmente, se integra todo el código anterior en un flujo de trabajo coherente que
aplica las distintas técnicas de interpretabilidad a una imagen de prueba del conjunto
CIFAR-10. El flujo incluye la carga de datos, la carga del modelo, la selección de una
muestra y la ejecución secuencial de los métodos de mapas de saliencia, Grad-CAM,
análisis por oclusión e Integrated Gradients. Guided Backpropagation está implementado y
puede utilizarse para construir Guided Grad-CAM si se desea extender el pipeline.

```python
def run_complete_pipeline() -> None:
    """
    Ejecuta de forma integrada todas las técnicas de interpretabilidad
    sobre una imagen de CIFAR-10.
    """
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETO: INTERPRETABILITY IN CNNs")
    print("=" * 70 + "\n")

    # Datos
    test_loader, _ = prepare_cifar10_data()

    # Modelo
    model = load_pretrained_model()

    # Seleccionar una imagen de prueba
    print("Seleccionando imagen de prueba...")
    images, labels = next(iter(test_loader))

    image = images[0:1]
    label = labels[0].item()
    original_image = denormalize_cifar10(images[0].clone())

    print(f"Clase real: {CIFAR10_CLASSES[label]}\n")

    # 1. Saliency Maps
    print("\n" + "=" * 70)
    print("EJECUTANDO: Saliency Maps")
    print("=" * 70 + "\n")

    saliency_gen = SaliencyMapGenerator(model, CONFIG["device"])
    saliency_gen.visualize_saliency(image.clone(), original_image)

    # 2. Grad-CAM
    print("\n" + "=" * 70)
    print("EJECUTANDO: Grad-CAM")
    print("=" * 70 + "\n")

    grad_cam = GradCAM(model, target_layer="layer4", device=CONFIG["device"])
    grad_cam.visualize_cam(image.clone(), original_image)

    # 4. Occlusion Analysis
    print("\n" + "=" * 70)
    print("EJECUTANDO: Occlusion Analysis")
    print("=" * 70 + "\n")

    occlusion = OcclusionAnalysis(model, device=CONFIG["device"])
    occlusion.visualize(image.clone(), original_image, patch_size=4, stride=2)

    # 5. Integrated Gradients
    print("\n" + "=" * 70)
    print("EJECUTANDO: Integrated Gradients")
    print("=" * 70 + "\n")

    ig = IntegratedGradients(model, device=CONFIG["device"])
    ig.visualize(image.clone(), original_image)


if __name__ == "__main__":
    run_complete_pipeline()
```

Este pipeline completo proporciona un marco práctico para explorar la interpretabilidad
en CNN sobre CIFAR-10. Aunque el modelo ResNet-18 no se ajusta explícitamente al conjunto
de datos en este script, la estructura de código permite reutilizar fácilmente el mismo
flujo de análisis con un modelo entrenado específicamente en CIFAR-10, simplemente
sustituyendo la función de carga del modelo por una versión que recupere pesos ajustados
al dominio de interés.
