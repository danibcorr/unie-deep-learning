## Copyright and License

© 2026, Daniel Bazo Correa

This course material is licensed under the MIT License.

### Disclaimer

- This course material is provided "as is", without warranty of any kind, express or
  implied.
- The author assumes no responsibility or liability for any errors, omissions, or
  outcomes resulting from the use of this content.
- All analyses and interpretations are for educational and research purposes only and do
  not constitute professional advice.

---

<p align="center">
  <a href="https://github.com/danibcorr/unie-deep-learning/actions/workflows/workflow.yaml"><img src="https://github.com/danibcorr/unie-deep-learning/actions/workflows/workflow.yaml/badge.svg"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue">
  <a href="https://github.com/danibcorr/unie-deep-learning/blob/main/LICENSE" target="_blank">
      <img src="https://img.shields.io/github/license/danibcorr/unie-deep-learning" alt="License">
  </a>
</p>

# Deep Learning Course

This repository contains a comprehensive deep learning course with hands-on examples
implemented in PyTorch. It provides a progressive introduction to core concepts,
mathematical foundations, and practical applications of deep learning, while remaining
accessible to non-specialist readers.

## Course Overview

The main topics covered in the course are:

- **Topic 1, Initial Concepts**: Introduction to neural networks, differences between
  traditional machine learning and deep learning, and main components of a deep learning
  workflow: Datasets, models, loss functions, optimization algorithms, and evaluation
  metrics.

- **Topic 2, Mathematics**: Mathematical foundations for deep learning, including linear
  algebra, calculus, and basic probability and statistics, with emphasis on their
  application to neural network training and tensor computation in PyTorch.

- **Topic 3, Applications**: End-to-end examples of deep learning applied to tasks such
  as classification, regression, and recommendation, including model definition, choice
  of loss functions and optimizers, and implementation of training and evaluation loops
  in PyTorch.

- **Topic 4, Computer Vision**: Deep learning methods for image processing and computer
  vision, including convolutional neural networks, image classification, object
  detection, segmentation, and transfer learning using pre-trained models.

- **Topic 5, Sequential Models**: Models for sequential data, such as text and
  time-ordered signals, covering recurrent architectures (RNNs, LSTMs, GRUs) and, where
  appropriate, attention mechanisms and transformer-based models.

- **Topic 6, Graph Models**: Introduction to graph neural networks and related
  architectures for node classification, link prediction, and graph classification, using
  graph-structured data.

## Quick Start

### Prerequisites

The following tools are required to install and run the course materials:

- [uv](https://github.com/astral-sh/uv) package manager: Tool used for fast and
  reproducible dependency and environment management.

### Installation

To obtain the materials and configure the environment, execute the following commands in
a terminal:

```bash
# Clone the repository
git clone https://github.com/danibcorr/unie-deep-learning.git
cd unie-deep-learning

# Install dependencies and set up the environment
make setup
```

The `make setup` command automatically installs the required Python dependencies and
prepares a suitable execution environment for the notebooks and scripts in the
repository.

## Author

This course has been developed by Daniel Bazo Correa. Additional information and
professional contact can be found through
[LinkedIn](https://www.linkedin.com/in/danibcorr/) or
[GitHub](https://github.com/danibcorr).
