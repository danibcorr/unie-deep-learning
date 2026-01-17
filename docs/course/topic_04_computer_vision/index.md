# Introduction

This topic systematically addresses the foundations and applications of _computer vision_
systems based on deep learning. The objective is to understand how to process and model
visual information, both in color image format (RGB format) and grayscale, and how these
representation decisions influence the models' ability to learn relevant patterns.

First, different image representation formats and their impact on subsequent processing
are studied. It analyzes when it is convenient to work with color images, leveraging
chromatic information, and when it is preferable to convert them to grayscale to simplify
the problem, reduce dimensionality, or focus attention on luminance structures. The
advantages and limitations of each approach and their relationship to the type of task
(classification, segmentation, detection, etc.) are discussed.

Next, various image processing techniques closely related to deep learning and, in many
cases, signal processing are introduced. It explains how to normalize input data, both at
the image and dataset level, and analyzes the influence of these normalizations on
numerical stability, convergence speed, and final model performance. On this basis,
normalization mechanisms applied within the architectures themselves are also studied,
such as _Batch Normalization_ and _Layer Normalization_, comparing their operating
principles, their implicit assumptions, and their effect when normalized based on data
statistics versus the use of internal model parameters.

A central block of the topic is dedicated to the study of **convolutional layers**. It
shows how to implement a convolution from scratch, analyzing in detail the filter sliding
operation over the image, the local linear combination of pixels, and the obtaining of
feature maps. Subsequently, it explains how to leverage primitives from libraries like
PyTorch to define and train convolutional layers efficiently, maintaining intuition about
what is happening at the mathematical and computational level.

In relation to convolutions, properties such as **translational invariance** and their
practical limitations are discussed with special attention. Although the ideal
convolution presents certain translational invariance, this property is altered when
strides greater than 1 or _pooling_ layers are introduced. In particular, it analyzes how
shifts (even circular ones) of the same image can produce significantly different
responses due to these mechanisms, and reflects on why, despite this, _pooling_ and
_stride_ are necessary to control spatial resolution, reduce computational cost, and
increase the effective receptive field of deep layers.

It is also emphasized that, even when convolutional layers do not require a fixed input
size, **image resolution** significantly conditions the type of features the network is
capable of learning. It discusses how the distribution of spatial frequencies and the
scale of present patterns (edges, textures, global structures) influence the model's
sensitivity to different levels of detail. This is essential to understand why the same
architecture can behave differently depending on the size and quality of input images.

From a historical perspective, the topic covers some of the **most influential
convolutional architectures**. It starts with LeNet, developed by Yann LeCun and his
team, used in a pioneering way in production environments for handwritten digit
recognition in bank checks, and continues with subsequent architectures that explore the
systematic increase in network depth and width. Models like VGG, which deepens the use of
homogeneous convolutional blocks, and ResNet, which introduces residual connections to
facilitate training of very deep networks, are analyzed. This review allows
contextualizing progress in _computer vision_ and understanding the motivations behind
the design of increasingly complex networks.

On this basis, **transfer learning techniques** are studied, which allow reusing
convolutional models pretrained by large organizations (such as Meta, Google, or others)
on massive datasets. It shows how to adapt these models to specific problems through
fine-tuning of the last layers, partial freezing of parameters, or redefinition of
classification heads, thus reducing training cost and improving performance in scenarios
with limited data.

In relation to efficiency and deployment on resource-constrained devices, methods such as
**knowledge distillation** are introduced, which consist of transferring the behavior of
large models (teachers) to smaller models (students). It discusses how these techniques
allow compressing convolutional networks and adapting them to embedded environments, such
as mobile phones, microcontrollers, or other edge computing systems, while maintaining a
significant fraction of their predictive capacity.

The topic also addresses **autoencoders**, both dense and convolutional, as tools for
unsupervised learning of representations from images. It analyzes how these architectures
allow extracting useful latent features for compression, anomaly detection, and
out-of-distribution sample detection. The basic encoder-decoder structure, the
reconstruction function, and the interpretation of latent variables are explained.

At a more advanced level, some **attention mechanisms applied to convolutional networks**
are presented, with special emphasis on _channel attention_. These mechanisms modulate
the relative importance of each feature map, allowing the network to focus on spatially
relevant patterns and enriching the extracted information without collapsing
representations. It discusses how these attention blocks can be integrated into standard
convolutional architectures and what benefits they provide in terms of performance and
robustness.

Subsequently, the application of **Transformer-type architectures to the visual domain**
is introduced, through models like Vision Transformer (ViT). The central idea of dividing
the image into patches, projecting them into an embedding space, and applying
self-attention mechanisms similar to those used in natural language processing is
explained. The original work proposed by Google is analyzed, as well as the implications
of transferring the global attention paradigm to the _computer vision_ context.

Finally, the field of **explainability and interpretability of vision models** is
addressed, with the aim of understanding why a model makes certain classification
decisions. Techniques based on weights and gradients are explored, as well as the
generation of heat maps that indicate the most relevant image regions for prediction.
Among these techniques is Grad-CAM and related methods, which allow visualizing the
effective attention of deep layers on the input image. These tools are essential for
auditing models, detecting biases, debugging unexpected behaviors, and increasing
confidence in systems deployed in sensitive environments.
