# Introduction

This topic studies **sequential models**, with special emphasis on their application to
text processing and time series. The starting point is the limitation of convolutional
architectures when applied directly to textual data: Although convolutions capture local
patterns efficiently, it is difficult for them to preserve and adequately model long
dependencies in a sequence. As text lengthens, part of the relevant contextual
information gradually dilutes or is lost, which limits these architectures' ability to
understand long-range relationships.

To address these limitations, models specifically designed for sequential data are
introduced. The first block focuses on **natural language processing (NLP)**, starting
with **tokenization** techniques. Tokenization transforms raw text into a sequence of
manageable units (tokens), which can be words, subwords, or characters. In this process,
input text cleaning is usually applied: punctuation marks, emojis, or other symbols are
removed or normalized according to the task, in order to obtain a more homogeneous
representation. On these tokens, a numerical representation that models can process is
subsequently built.

Once the text is tokenized, the use of **embedding layers** is introduced, which allow
mapping each token to a dense vector in a fixed-dimensional space. These embeddings can
be learned from scratch during model training, or initialized from **pretrained
dictionaries** and **already trained tokenizers**, leveraging prior knowledge accumulated
in large corpora. In this context, the advantages and disadvantages of both strategies
are analyzed: learning task-specific embeddings versus reuse and fine-tuning of
pre-existing embeddings.

On these vector representations, the first classical sequential architectures are
studied, based on **recurrent neural networks (RNN)**. RNNs process the sequence step by
step, maintaining a hidden state that acts as memory of what has already been seen.
However, they present important limitations, particularly the **vanishing gradient
problem**: When sequences are long or the network is deep, gradients that propagate
toward distant time steps tend to become very small, hindering the learning of long-term
dependencies and making training unstable or inefficient.

To mitigate these problems, **LSTM (Long Short-Term Memory)** are introduced, a variant
of recurrent networks that incorporates **memory cells** and _gate_ mechanisms. These
gates explicitly control what information is stored, what is forgotten, and what is
exposed at each time step, allowing information to remain relevant for longer intervals.
Their internal structure, the role of input, forget, and output gates, and the advantages
they provide over simple RNNs are analyzed.

In addition to text, sequential models are naturally applied to **time series**. In this
context, **autoencoders** and recurrent or convolutional variants are presented as tools
for unsupervised learning of temporal patterns. These models allow detecting
**anomalies** and unusual patterns by comparing the observed signal with the
reconstruction produced by the decoder, as well as identifying samples that are _out of
distribution_. The basic encoder-decoder configuration, the reconstruction function, and
the use of error thresholds for decision-making are discussed.

Next, the architectures that constitute the **state of the art in 2025** for sequential
processing are introduced: **Transformers**. These models are based on **attention
mechanisms**, which allow each position in the sequence to relate directly to any other,
capturing dependencies at multiple scales without resorting to explicit recurrences. A
decisive advantage of Transformers is their capacity for **parallel processing** of
tokens during the training phase, which enables very efficient exploitation of
accelerated hardware (GPUs, TPUs) and overcomes one of the main limitations of RNNs and
LSTMs, whose sequential processing hinders parallelization and slows convergence.

On the basis of the standard Transformer, different **derived architectures and
extensions** are studied, including **Mixture of Experts (MoE)** approaches. In these
models, multiple experts are trained (for example, several Transformer-type models or
related variants) and their predictions are combined through a routing module (_gating
network_) that decides which experts to activate for each input. This scheme allows
effectively increasing model capacity while maintaining controlled computational cost, by
activating only a subset of experts for each input.

Finally, the use of **pretrained models** in the sequential domain is explored, both for
language and multimodal, with special attention to those with **open weights** or whose
architectures have been released by the research community and industry. The
**Transformers library from Hugging Face** is presented as a standard tool for loading,
using, and adapting pretrained models, both language models (LLMs of different sizes) and
vision and multimodal models. It shows how to integrate pretrained tokenizers, how to
reuse embeddings, and how to perform _fine-tuning_ or _inference-only_ tasks with
relatively low implementation effort.
