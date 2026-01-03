# Introduction

In this topic, we study **Graph Neural Networks (GNN)**, whose main objective is to learn
from data that can be naturally represented as **graphs**. A graph is a structure formed
by **nodes (vertices)** and **edges** that describe relationships or interactions between
entities. This form of representation is especially suitable for numerous real-world
problems, in which the connections between elements are as important as the elements
themselves.

Graphs appear implicitly in many everyday domains. In **social networks**, for example,
each user can be modeled as a node and friendship, following, or interaction
relationships as edges. Furthermore, attributes or characteristics can be associated with
each node, such as salary range, age, geographic location, occupation, or demographic
variables. These characteristics, together with the connection structure, allow building
models that better capture the dynamics and organization of the social network than if
users were considered in isolation.

Similarly, an **image** can be interpreted as a graph in which each pixel acts as a node
connected to its neighboring pixels (in a 4, 8, or wider neighborhood), and where the
intensity or color vector (for example, in RGB format) is used as node attributes. This
graph perspective allows applying graph-based processing techniques to visual problems,
establishing a bridge between _computer vision_ and _graph learning_.

There are also graph representations that are especially relevant in scientific and
industrial fields. A paradigmatic case is that of **molecules**, where atoms are modeled
as nodes and chemical bonds as edges. This type of representation has been key in
developments such as AlphaFold, presented by Google DeepMind, for protein folding, or in
multiple works on _drug discovery_ and computational chemistry. Another example is
offered by systems like **Google Maps**, where intersections and points of interest can
be considered nodes, and roads or connections between them, edges; on this structure,
problems such as traffic prediction, travel time estimation, or optimal route calculation
are formulated.

Throughout the topic, we will see how to **preprocess and transform heterogeneous data**
to represent them as graphs suitable for training neural models. We will analyze how to
explicitly define nodes, edges, and their attributes, and how to construct data
structures compatible with modern deep learning libraries for graphs.

A central objective is to understand how to **implement a graph from scratch** and how to
apply different variants of graph neural networks to it. Special emphasis is placed on
**GNN based on graph convolutions** (_Graph Convolutional Networks, GCN_), which
generalize the idea of classical convolution to non-Euclidean domains. In this framework,
each node updates its representation by aggregating information from its neighbors
according to a _message passing_ scheme, which allows capturing local and global patterns
in the graph structure.

In addition to graph convolutions, we also study **attention mechanisms in graphs**
(_Graph Attention Networks, GAT_ and variants). These methods assign differentiated
weights to connections between nodes, allowing the network to learn which neighbors it
should pay more attention to depending on the context and task. This approach is
especially useful when the importance of relationships is not homogeneous or when a finer
interpretation of interactions between nodes is desired.

The techniques described are applied both to **images modeled as graphs** and to other
types of data structured in networks, demonstrating the versatility of GNNs to address
problems of node classification, link prediction, whole graph classification, and other
related tasks.

For practical implementation, we will use the **PyTorch Geometric** library, a
specialized framework that extends PyTorch with data structures and optimized operations
for deep learning on graphs. PyTorch Geometric facilitates the definition of graphs, the
construction of graph convolution and attention layers, and the training of complex
models on large collections of graphs or large-scale graphs.
