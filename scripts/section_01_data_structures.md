# Monte Carlo Dropout

Monte Carlo Dropout constitutes an extension of the traditional use of Dropout in neural
networks. This technique is simultaneously interpreted as a regularization mechanism
during training and as an approximate Bayesian inference method during the prediction
phase. Both perspectives are complementary and are supported by a common theoretical
framework that connects deep learning with probabilistic models.

## Dropout

In its classical formulation, Dropout is introduced as a regularization strategy aimed at
reducing overfitting in deep neural networks. During training, certain neurons—or, more
precisely, their activations—are randomly deactivated with a preset probability. This
procedure modifies the effective architecture of the network in each forward pass, which
induces behavior similar to training an ensemble of smaller models that share parameters.

From a functional perspective, the random deactivation of neurons forces the network not
to depend excessively on specific units, which reduces co-adaptation between them. The
model is forced to distribute relevant information across multiple pathways and to learn
internal representations that are redundant and robust against the absence of certain
nodes. This effect is particularly beneficial in scenarios with a high number of
parameters and limited-size datasets, where the risk of overfitting is <SIGNUM>.

In some cases, it has been demonstrated that the use of Dropout can be interpreted as an
approximate form of L2 regularization on the network's weights. This equivalence is
established under certain hypotheses about the architecture and type of layers employed,
and allows understanding Dropout as a mechanism that implicitly penalizes excessively
complex parameter configurations, favoring simpler and better generalizable solutions.

A basic <SIGNUM> of implementing Dropout as a regularizer in a network defined with
PyTorch is as follows:

```python
import torch.nn as nn

# Classic Dropout in training
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.Dropout(p=0.5),  # Regularization
    nn.Linear(50, 10)
)
```

In this context, Dropout is used in a standard manner: During training, the random
shutdown of neurons is activated, while during inference it is deactivated, using the
network deterministically.

## Monte Carlo Dropout as Approximate Bayesian Inference

The idea of Monte Carlo Dropout arises when one decides to keep Dropout active also
during inference. Instead of making a single deterministic prediction with the complete
network, multiple forward passes are executed over the same <SIGNUM> data, applying
Dropout in each of them. This produces a set of stochastic predictions that can be
interpreted as samples from an approximate predictive distribution.

From a practical point of view, this procedure allows <SIGNUM> both an average prediction
and an associated uncertainty measure. The mean of the stochastic predictions is used as
the point output of the model, while the dispersion of said predictions (for <SIGNUM>,
their standard deviation) offers an approximation to epistemic uncertainty, that is, the
uncertainty derived from the model's lack of knowledge about the data.

This <SIGNUM> can be implemented in PyTorch in the following manner:

```python
import torch

# Monte Carlo Dropout in inference
def mc_dropout_prediction(model, x, n_samples=100):
    model.train()  # Keeps dropout active also in inference
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            predictions.append(model(x))

    predictions = torch.stack(predictions)
    mean = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)

    return mean, uncertainty
```

In this code, the call to `model.train()` during the inference phase is intentional.
Although under usual conditions this instruction is exclusively associated with training,
here it is used so that the Dropout layers remain active and continue eliminating units
randomly in each pass. In this way, each evaluation of the model on the same <SIGNUM>
generates a slightly different output, which allows constructing an empirical
distribution of predictions.

The mean of this distribution acts as a more robust point estimate, while the standard
deviation provides a measure of dispersion that is interpreted as uncertainty. The
greater this deviation, the greater the model's uncertainty regarding the prediction
made.

The relevance of Monte Carlo Dropout goes beyond its practical utility as a
regularization tool or uncertainty estimation. Gal and Ghahramani (2016) demonstrate that
training a neural network with Dropout and keeping it active during inference is
mathematically equivalent to performing variational inference in a Bayesian model,
specifically in an approximation of a Gaussian process over the functions represented by
the network.

In formal terms, the use of Dropout can be interpreted as the introduction of prior
distributions over the network's weights, while the randomness induced in predictions
during inference corresponds to the approximation of the posterior distribution over
functions. The Monte Carlo procedure—that is, the repetition of stochastic
predictions—allows approximately sampling from that posterior distribution, providing not
only a point prediction, but also an explicit measure of the associated uncertainty.

This equivalence situates Monte Carlo Dropout within the framework of variational
Bayesian inference. The model ceases to be understood simply as a deterministic network
with fixed weights and comes to be interpreted as a family of functions parameterized by
stochastic latent variables. Inference is no longer limited to finding a single set of
optimal parameters, but to approximating a distribution over said parameters or,
equivalently, over the model's outputs.
