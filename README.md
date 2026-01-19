<p align="center">
  <img src="./docs/assets/images/logo.png" height="250"/>
  <br/>
</p>

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

## Documentation

The full course documentation is available at:  
[https://danibcorr.github.io/unie-deep-learning/](https://danibcorr.github.io/unie-deep-learning/)

To serve the documentation locally from the project root, run:

```bash
make doc
```

This command generates or updates the documentation and starts a local web server. The
URL to access the documentation is displayed in the terminal output.

## License

This project is distributed under the MIT License. The complete license text is available
in the [LICENSE](LICENSE) file included in this repository.

## Author

This course has been developed by Daniel Bazo Correa. Additional information and
professional contact can be found through LinkedIn at
[@danibcorr](https://www.linkedin.com/in/danibcorr/) or GitHub at
[@danibcorr](https://github.com/danibcorr).
