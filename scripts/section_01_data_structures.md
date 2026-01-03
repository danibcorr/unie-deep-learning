# Data Structures and Tensors in PyTorch

In this section, the fundamental data structures of PyTorch are presented, with special
emphasis on the concept of tensor, its creation, its main properties, and its central
role in deep learning. The purpose is to understand what a tensor is, how it is
constructed, what types exist, and why it is essential both in the design and
experimentation phases as well as during model training and inference.

Tensors constitute the basic data structure of PyTorch and can be considered a
generalization of scalars, vectors, and matrices to an arbitrary number of dimensions.
Each tensor stores numerical values ordered according to a determined shape, and can
reside in CPU or GPU. In practice, tensors are analogous to NumPy's `ndarray`, but are
optimized for intensive numerical computation and for acceleration through specialized
hardware, such as graphics processing units (GPUs).

Throughout this section, it is shown how to create tensors from predefined lists or
matrices, how to choose their data type (for example, integers or floats with different
precisions), and how to query and modify their basic properties, such as the number of
dimensions (`ndim`), the shape (`shape`), or the computing device on which they are
stored (`device`). These aspects are essential to ensure that operations between tensors
are compatible and executed efficiently.

## GPU Availability

In the context of machine learning, the use of GPUs allows significantly accelerating
tensor processing and the execution of large-scale neural models. PyTorch provides
utilities to verify if the system has a compatible GPU and to obtain information about
available devices. The following code snippet illustrates how to perform this check:

```python
# 3pps
import torch

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    print("GPU available")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU detected, CPU will be used")
```

This type of check is especially useful at the beginning of a Jupyter notebook or
training script, as it allows dynamically adapting the code to available hardware, moving
tensors and models to the appropriate device through operations like `tensor.to("cuda")`
or `model.to("cuda")`.

### Performance Comparison: CPU vs GPU

To illustrate the computational advantages of using GPUs, consider the following example
that compares the execution time of matrix multiplication operations on both devices:

```python
import time

# Define matrix size
size = (5000, 5000)

# Create tensors on CPU and GPU
cpu_tensor = torch.rand(size)
if torch.cuda.is_available():
    gpu_tensor = torch.rand(size, device='cuda')

    # Measure CPU time
    start = time.time()
    cpu_result = cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")

    # Measure GPU time
    start = time.time()
    gpu_result = gpu_tensor @ gpu_tensor
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
```

The `torch.cuda.synchronize()` call is essential to ensure accurate timing measurements,
as GPU operations are asynchronous by default.

## Introduction to Tensors

![image.png](attachment:image.png)

A tensor is the fundamental data structure in PyTorch. From an intuitive point of view, a
scalar is a 0-dimensional tensor, a vector is a 1-dimensional tensor, a matrix is a
2-dimensional tensor, and from there on, we speak of higher-order tensors (3D, 4D, etc.).
This generalization allows representing very diverse data, such as images, tokenized text
sequences, or multivariate time series, in a unified way.

Tensors allow storing data efficiently, both in CPU and GPU, and support a wide variety
of mathematical operations: additions, products, reductions, linear algebra operations,
and many others. Most deep learning algorithms are implemented as compositions of
operations on tensors.

### PyTorch Tensor Data Types

PyTorch supports various data types optimized for different use cases. The following
table summarizes the most common types:

| dtype                               | Description                   | Typical Use Case                                   | Memory per Element |
| ----------------------------------- | ----------------------------- | -------------------------------------------------- | ------------------ |
| `torch.float32` (or `torch.float`)  | 32-bit floating point         | Default type, general purpose training             | 4 bytes            |
| `torch.float64` (or `torch.double`) | 64-bit floating point         | High precision scientific computing                | 8 bytes            |
| `torch.float16` (or `torch.half`)   | 16-bit floating point         | Mixed precision training, inference acceleration   | 2 bytes            |
| `torch.bfloat16`                    | Brain floating point (16-bit) | Modern TPU/GPU training, better range than float16 | 2 bytes            |
| `torch.int64` (or `torch.long`)     | 64-bit integer                | Indices, labels, sizes                             | 8 bytes            |
| `torch.int32` (or `torch.int`)      | 32-bit integer                | Integer computations                               | 4 bytes            |
| `torch.int16` (or `torch.short`)    | 16-bit integer                | Memory-constrained integer storage                 | 2 bytes            |
| `torch.int8`                        | 8-bit integer                 | Quantized models, extreme memory savings           | 1 byte             |
| `torch.uint8`                       | 8-bit unsigned integer        | Image data (0-255 range)                           | 1 byte             |
| `torch.bool`                        | Boolean                       | Masks, logical conditions                          | 1 byte             |

The choice of data type significantly impacts memory consumption, computational speed,
and numerical stability. For instance, using `float16` can reduce memory usage by 50%
compared to `float32`, enabling training of larger models, but may require careful
handling of numerical underflow/overflow.

### Creating Tensors: Scalars, Vectors, Matrices, and Higher-Order Tensors

PyTorch facilitates the creation of tensors of different dimensions. The following code
snippet illustrates how to construct a scalar, a vector, a matrix, and a higher-order
tensor:

```python
# Scalar tensor
scalar = torch.tensor(7)
scalar
```

A scalar has no additional dimensions, so its number of dimensions is 0:

```python
scalar.ndim
```

To obtain the Python numerical value associated with the scalar, the `.item()` method is
used:

```python
scalar.item()
```

From there, vectors (1-dimensional tensors) can be defined by providing a list of values:

```python
# Creating a vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)
```

In this case, `vector.ndim` returns `1`, and `vector.shape` indicates the vector's
length. Similarly, a matrix is represented as a list of lists, generating a 2-dimensional
tensor:

```python
# Creating a matrix
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)
```

The above matrix has two rows and three columns, so its shape is `(2, 3)`.

Higher-order tensors are constructed by nesting additional lists. For example, the
following tensor has three dimensions, organized hierarchically:

```python
# Creating a three-dimensional tensor
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [6, 4, 3]]])
print(tensor)
print(tensor.ndim)
print(tensor.shape)
```

The `shape` attribute describes the size of each dimension. Understanding this structure
is key to designing and interpreting neural network architectures, as the inputs and
outputs of each layer are represented as tensors with specific shapes.

### Creating Tensors with Values

In practice, it is very common to create test tensors or initialize model parameters
using random values. PyTorch allows creating tensors with values generated randomly from
different distributions. For example:

```python
# Random tensors
random_tensor = torch.rand((2, 3, 4))
print(random_tensor)
random_tensor.ndim, random_tensor.shape
```

Here a tensor with shape `(2, 3, 4)` is created whose elements are uniformly distributed
in the interval `[0, 1)`. This type of tensor is useful for:

1. Verifying that input and output dimensions of a model are coherent between layers.
2. Checking that internal operations are performed without errors before using real data.
3. Exploring model behavior in unit tests or quick experiments.

In addition to random tensors, it is common to use tensors initialized with zeros or
ones, for example, to define masks, templates, or initial values:

```python
zero_tensors = torch.zeros((3, 4))
zero_tensors
```

```python
ones_tensors = torch.ones((3, 4))
ones_tensors
```

You can also create tensors containing sequences of equally spaced values using
`torch.arange`:

```python
range_tensor = torch.arange(start=0, end=100, step=2)
print(range_tensor)
print(f"Shape of 'range_tensor': {range_tensor.shape}")
```

This tensor contains values from 0 to 98 with a step of 2. From it, other tensors that
inherit its shape can be constructed:

```python
# Create a tensor with the same dimension as another tensor
range_copy = torch.zeros_like(input=range_tensor)
print(f"Shape of 'range_copy': {range_copy.shape}")
range_copy
```

The use of functions like `zeros_like` or `ones_like` facilitates creating tensors
compatible in shape and data type with existing ones.

### Conversion Between NumPy and PyTorch

PyTorch tensors and NumPy arrays are closely related, and conversion between them is
straightforward and efficient. This interoperability is crucial when integrating PyTorch
with other scientific computing libraries.

```python
import numpy as np

# NumPy array to PyTorch tensor
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Original NumPy array:\n{numpy_array}")
print(f"Converted tensor:\n{tensor_from_numpy}")
print(f"Tensor dtype: {tensor_from_numpy.dtype}")

# PyTorch tensor to NumPy array
tensor = torch.tensor([[7, 8, 9], [10, 11, 12]])
numpy_from_tensor = tensor.numpy()
print(f"\nOriginal tensor:\n{tensor}")
print(f"Converted NumPy array:\n{numpy_from_tensor}")
```

**Important considerations:**

1. **Memory sharing**: By default, `torch.from_numpy()` creates a tensor that shares
   memory with the original NumPy array. Modifications to one will affect the other.

```python
# Demonstrate memory sharing
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
np_array[0] = 999
print(f"Modified NumPy array: {np_array}")
print(f"Tensor (shares memory): {tensor}")  # Also shows 999
```

2. **GPU tensors**: Tensors on GPU must be moved to CPU before converting to NumPy:

```python
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
    # This would raise an error: gpu_tensor.numpy()
    numpy_from_gpu = gpu_tensor.cpu().numpy()  # Correct approach
```

3. **Gradient tracking**: Tensors with gradients enabled must have gradients detached:

```python
tensor_with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# This would raise an error: tensor_with_grad.numpy()
numpy_array = tensor_with_grad.detach().numpy()  # Correct approach
```

### Data Types, Devices, and Tensor Compatibility

Each tensor in PyTorch is characterized, among other aspects, by its data type (`dtype`),
its shape (`shape`), and the computing device on which it resides (`device`). These
attributes influence operation compatibility and computational performance. When
operations are performed between tensors whose data types do not match, whose dimensions
are not compatible for the defined operation, or that are on different devices (for
example, one on CPU and another on GPU), conflicts and errors can occur during execution.

The following snippet shows how to inspect these properties:

```python
tensor = torch.rand(size=(2, 2, 3))
tensor
```

```python
print(f"Data type: {tensor.dtype}")
print(f"Shape: {tensor.shape}")
print(f"Device: {tensor.device}")
```

By default, floating-point tensors are created with type `torch.float32`. However, it is
possible to specify a different data type, such as `float16` or `float64`:

```python
tensor = torch.rand(size=(2, 2, 3), dtype=torch.float16)
tensor
```

```python
print(f"Data type: {tensor.dtype}")
print(f"Shape: {tensor.shape}")
print(f"Device: {tensor.device}")
```

The choice of numerical precision involves a trade-off between computational cost, memory
consumption, training stability, and result accuracy. For example, `float16` allows
significantly accelerating training and inference on appropriate hardware, but may
increase the risk of numerical stability problems in certain models.

In general, it is important that tensors involved in the same operation share a
compatible data type and are on the same device. Otherwise, it is necessary to explicitly
convert the type (`tensor.to(torch.float32)`, `tensor.int()`, etc.) or move the tensor to
the corresponding device (`tensor.to("cuda")`, `tensor.to("cpu")`).

### Common Errors and Best Practices

When working with tensors, certain errors frequently arise. Understanding these common
pitfalls helps prevent debugging headaches:

#### 1. Incompatible Dimensions

```python
# Error: dimension mismatch
try:
    a = torch.rand(2, 3)
    b = torch.rand(3, 5)
    c = a + b  # Raises error: shapes [2, 3] and [3, 5] are incompatible
except RuntimeError as e:
    print(f"Error: {e}")

# Correct: use matrix multiplication for different shapes
c = torch.matmul(a, b)  # Results in shape [2, 5]
print(f"Correct result shape: {c.shape}")
```

#### 2. Device Mismatch

```python
# Error: tensors on different devices
if torch.cuda.is_available():
    try:
        cpu_tensor = torch.rand(2, 3)
        gpu_tensor = torch.rand(2, 3, device='cuda')
        result = cpu_tensor + gpu_tensor  # Raises error
    except RuntimeError as e:
        print(f"Error: {e}")

    # Correct: ensure both tensors are on the same device
    result = cpu_tensor.to('cuda') + gpu_tensor
    # Or: result = cpu_tensor + gpu_tensor.to('cpu')
```

#### 3. Data Type Incompatibility

```python
# Error: incompatible data types
try:
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    # Some operations may fail or produce unexpected results
    result = int_tensor / float_tensor  # Works but with implicit conversion
except Exception as e:
    print(f"Error: {e}")

# Best practice: explicit type conversion
result = int_tensor.float() / float_tensor
print(f"Result dtype: {result.dtype}")
```

#### 4. Gradient-Related Issues

```python
# Error: modifying tensors with gradients in-place
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

try:
    x[0] = 10.0  # In-place modification raises error
except RuntimeError as e:
    print(f"Error: {e}")

# Correct: use operations that don't modify tensors in-place
x_new = torch.tensor([10.0, 2.0, 3.0], requires_grad=True)
y = x_new ** 2
```

#### 5. Memory Leaks with Gradients

```python
# Potential memory leak: accumulating gradients in a loop
losses = []
for i in range(100):
    x = torch.randn(1000, 1000, requires_grad=True)
    y = x.sum()
    # losses.append(y)  # BAD: keeps entire computational graph
    losses.append(y.item())  # GOOD: only stores the scalar value
```

### Basic Operations

Once tensors are created, PyTorch allows applying various reduction and aggregation
operations on them. For example, you can calculate maximums, means, or maximum indices
along specific dimensions. Consider the following two-dimensional tensor:

```python
tensor = torch.rand(size=(2, 3))
tensor
```

The `max` method allows obtaining the maximum value along a dimension and its index:

```python
# Maximum by columns (dimension 0)
tensor.max(dim=0)
```

```python
# Maximum by rows (dimension 1)
tensor.max(dim=1)
```

In this context, the convention is adopted that columns correspond to axis or dimension
`0`, while rows are associated with axis or dimension `1`. Similarly, the mean can be
calculated:

```python
tensor.mean(dim=0), tensor.mean(dim=1)
```

```python
torch.mean(tensor, dim=0), torch.mean(tensor, dim=1)
```

The `argmax` function returns the indices of maximum values along a determined dimension:

```python
torch.argmax(tensor, dim=0), tensor.argmax(dim=0)
```

```python
tensor.argmax(dim=1)
```

These indices can be used to select elements or substructures of the tensor. For example:

```python
tensor
```

```python
tensor[:, tensor.argmax(dim=1)[0]]
```

Here all rows and the column corresponding to the largest value of the first row are
selected, illustrating how to combine reduction operations with indexing.

### Mathematical Operations Between Tensors

PyTorch supports a comprehensive set of mathematical operations between tensors.
Understanding the differences between element-wise operations, matrix multiplication, and
broadcasting is fundamental.

#### Element-wise Operations

Element-wise operations apply to corresponding elements of tensors with compatible
shapes:

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Element-wise addition
print("Addition (a + b):")
print(a + b)

# Element-wise subtraction
print("\nSubtraction (a - b):")
print(a - b)

# Element-wise multiplication
print("\nElement-wise multiplication (a * b):")
print(a * b)

# Element-wise division
print("\nElement-wise division (a / b):")
print(a / b)

# Element-wise power
print("\nPower (a ** 2):")
print(a ** 2)
```

#### Matrix Multiplication

PyTorch provides multiple ways to perform matrix multiplication, each with specific use
cases:

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Method 1: torch.matmul (recommended, works with batched matrices)
result1 = torch.matmul(a, b)
print("torch.matmul(a, b):")
print(result1)

# Method 2: @ operator (syntactic sugar for matmul)
result2 = a @ b
print("\na @ b:")
print(result2)

# Method 3: torch.mm (only for 2D matrices)
result3 = torch.mm(a, b)
print("\ntorch.mm(a, b):")
print(result3)

# Verify all methods give the same result
print(f"\nAll methods equivalent: {torch.equal(result1, result2) and torch.equal(result2, result3)}")
```

**Key differences:**

- `torch.matmul` (or `@`): Most versatile, handles broadcasting and batched operations
- `torch.mm`: Only for 2D matrices, slightly faster but less flexible
- `*`: Element-wise multiplication (NOT matrix multiplication)

#### Broadcasting

Broadcasting allows operations between tensors of different shapes by automatically
expanding dimensions:

```python
# Example 1: Scalar and tensor
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = 10
result = a + b  # b is broadcasted to match a's shape
print("Tensor + Scalar:")
print(result)

# Example 2: Vector and matrix
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
b = torch.tensor([10, 20, 30])  # Shape: (3,)
result = a + b  # b is broadcasted along dim 0
print("\nMatrix + Vector:")
print(result)

# Example 3: Broadcasting with unsqueeze
a = torch.tensor([[1], [2], [3]])  # Shape: (3, 1)
b = torch.tensor([10, 20])  # Shape: (2,)
result = a + b  # Broadcasted to (3, 2)
print("\nBroadcasting with different dimensions:")
print(result)
print(f"Result shape: {result.shape}")
```

**Broadcasting rules:**

1. Dimensions are aligned from right to left
2. Dimensions must be equal, one of them must be 1, or one doesn't exist
3. The result has the maximum size along each dimension

#### In-place Operations

Operations ending with an underscore (`_`) modify tensors in-place, saving memory but
potentially causing issues with gradient computation:

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Original x: {x}")
print(f"Memory address: {x.data_ptr()}")

# In-place addition
x.add_(5)
print(f"After add_(5): {x}")
print(f"Memory address: {x.data_ptr()}")  # Same address

# In-place multiplication
x.mul_(2)
print(f"After mul_(2): {x}")

# Compare with non-in-place operations
y = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"\nOriginal y: {y}")
print(f"Memory address: {y.data_ptr()}")

y = y + 5  # Creates new tensor
print(f"After y = y + 5: {y}")
print(f"Memory address: {y.data_ptr()}")  # Different address
```

**Warning:** Avoid in-place operations on tensors with `requires_grad=True` as they can
cause errors during backpropagation.

### Tensor Reshaping and Manipulation

Understanding how to reshape and manipulate tensors is crucial for building neural
networks, as layer outputs often need to be reshaped to match subsequent layer inputs.

#### View, Reshape, and Flatten

```python
# Create a sample tensor
x = torch.arange(12)
print(f"Original tensor: {x}")
print(f"Shape: {x.shape}")

# Using view (shares memory with original tensor)
x_view = x.view(3, 4)
print(f"\nView as (3, 4):\n{x_view}")

# Using reshape (may create a copy if necessary)
x_reshape = x.reshape(2, 6)
print(f"\nReshape as (2, 6):\n{x_reshape}")

# Using -1 for automatic dimension calculation
x_auto = x.view(3, -1)  # -1 automatically becomes 4
print(f"\nView as (3, -1):\n{x_auto}")

# Flatten: convert to 1D
x_flat = x.view(-1)
print(f"\nFlattened: {x_flat}")

# Flatten with method
x_flat2 = x.flatten()
print(f"Flatten method: {x_flat2}")
```

**Key differences:**

- `view()`: Returns a view of the original tensor (shares memory), requires contiguous
  memory
- `reshape()`: May return a view or copy, more flexible but potentially less efficient
- `flatten()`: Always returns a 1D tensor

#### Memory Contiguity

```python
# Demonstrate the difference between view and reshape with transpose
x = torch.arange(12).reshape(3, 4)
print(f"Original:\n{x}")
print(f"Is contiguous: {x.is_contiguous()}")

# Transpose makes tensor non-contiguous
x_t = x.t()
print(f"\nTransposed:\n{x_t}")
print(f"Is contiguous: {x_t.is_contiguous()}")

# view() fails on non-contiguous tensor
try:
    x_t.view(12)
except RuntimeError as e:
    print(f"\nError with view(): {e}")

# reshape() works (creates a copy)
x_t_reshaped = x_t.reshape(12)
print(f"\nReshape works: {x_t_reshaped}")

# Make contiguous explicitly
x_t_cont = x_t.contiguous()
x_t_view = x_t_cont.view(12)
print(f"After contiguous(), view works: {x_t_view}")
```

#### Squeeze and Unsqueeze

These operations add or remove dimensions of size 1:

```python
# Create a tensor with dimensions of size 1
x = torch.rand(1, 3, 1, 4)
print(f"Original shape: {x.shape}")

# Remove all dimensions of size 1
x_squeezed = x.squeeze()
print(f"After squeeze(): {x_squeezed.shape}")

# Remove specific dimension
x_squeezed_dim = x.squeeze(dim=0)
print(f"After squeeze(dim=0): {x_squeezed_dim.shape}")

# Add dimension
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"After unsqueeze(dim=0): {x_unsqueezed.shape}")

# Practical example: preparing batch dimension
single_image = torch.rand(3, 224, 224)  # C, H, W
batched_image = single_image.unsqueeze(0)  # Add batch dimension: B, C, H, W
print(f"\nSingle image shape: {single_image.shape}")
print(f"Batched image shape: {batched_image.shape}")
```

### Concatenation and Stacking

Combining multiple tensors is a common operation in deep learning:

```python
# Create sample tensors
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
c = torch.full((2, 3), 0.5)

print("Tensor a:")
print(a)
print("\nTensor b:")
print(b)
print("\nTensor c:")
print(c)

# Concatenation along dimension 0 (rows)
cat_dim0 = torch.cat([a, b, c], dim=0)
print(f"\nConcatenate along dim=0:")
print(cat_dim0)
print(f"Shape: {cat_dim0.shape}")

# Concatenation along dimension 1 (columns)
cat_dim1 = torch.cat([a, b, c], dim=1)
print(f"\nConcatenate along dim=1:")
print(cat_dim1)
print(f"Shape: {cat_dim1.shape}")

# Stacking creates a new dimension
stacked = torch.stack([a, b, c], dim=0)
print(f"\nStack along dim=0:")
print(stacked)
print(f"Shape: {stacked.shape}")

# Stack along different dimension
stacked_dim1 = torch.stack([a, b, c], dim=1)
print(f"\nStack along dim=1 shape: {stacked_dim1.shape}")
```

**Key differences:**

- `torch.cat()`: Concatenates along an existing dimension
- `torch.stack()`: Creates a new dimension and stacks along it

#### Splitting Tensors

```python
# Create a tensor to split
x = torch.arange(24).reshape(4, 6)
print("Original tensor:")
print(x)
print(f"Shape: {x.shape}")

# Split into equal chunks
chunks = torch.split(x, split_size_or_sections=2, dim=0)
print(f"\nSplit into chunks of size 2 along dim=0:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} shape: {chunk.shape}")
    print(chunk)

# Split into specific sizes
split_sizes = [1, 2, 3]
custom_chunks = torch.split(x, split_sizes, dim=1)
print(f"\nSplit into custom sizes {split_sizes} along dim=1:")
for i, chunk in enumerate(custom_chunks):
    print(f"Chunk {i} shape: {chunk.shape}")

# Chunk: split into specified number of chunks
num_chunks = 2
chunked = torch.chunk(x, chunks=num_chunks, dim=0)
print(f"\nChunk into {num_chunks} parts:")
for i, chunk in enumerate(chunked):
    print(f"Chunk {i} shape: {chunk.shape}")
```

### Indexing

PyTorch also allows performing advanced indexing operations and constructing submatrices
through slicing techniques. Consider a random matrix of size `(4, 4)`:

```python
matrix = torch.rand((4, 4))
matrix
```

It is possible to extract submatrices taking one element out of every two in both
dimensions:

```python
submatrix_1 = matrix[0::2, 0::2]
submatrix_2 = matrix[0::2, 1::2]
submatrix_3 = matrix[1::2, 0::2]
submatrix_4 = matrix[1::2, 1::2]

submatrix_1, submatrix_2, submatrix_3, submatrix_4
```

These submatrices can be stacked along a new dimension using `torch.stack`:

```python
submatrices = torch.stack([submatrix_1, submatrix_2, submatrix_3, submatrix_4])
submatrices
```

```python
submatrices.shape
```

The result is a tensor in which each submatrix occupies a position along the first
dimension. If you want to add an additional dimension, you can use `unsqueeze`:

```python
print(submatrices.shape)
submatrices = submatrices.unsqueeze(dim=0)
submatrices.shape
```

From this tensor, different operations can be performed. For example, we will calculate
the Frobenius matrix norm (the Frobenius norm is equivalent to the square root of the sum
of squares of all matrix elements) of each submatrix using `torch.linalg.matrix_norm`:

```python
norm = torch.linalg.matrix_norm(submatrices, ord="fro", dim=(-2, -1))
norm
```

Once the norms are calculated, you can select the submatrix with the highest norm using
`argmax` on the `norm` tensor:

```python
submatrices[:, torch.argmax(norm), :, :]
```

This example illustrates how to combine indexing, stacking, dimension insertion, and
linear algebra operations to analyze and manipulate complex matrix structures within a
tensor.

### Reproducibility

Reproducibility constitutes a fundamental requirement in the development and evaluation
of machine learning models. In this context, reproducibility is understood as the ability
to obtain the same results when repeatedly executing an experiment under the same
conditions: same code, same data, same hyperparameter configuration and, especially
relevant, same random initialization.

In PyTorch, an important part of model behavior depends on random processes, such as the
initialization of neural network weights, the generation of tensors with random values,
or random data sampling during training. If these processes are not controlled, small
variations in initializations can produce different results in each execution, making it
difficult to compare experiments and debug errors.

To mitigate this problem, PyTorch provides mechanisms that allow fixing the seed of the
random number generator. One of the most used is the instruction:

```python
torch.manual_seed(42)
```

This call initializes PyTorch's random number generator with a fixed seed, in this case
the value `42`. From that moment on, all random operations that depend on this generator
will produce the same sequence of values in successive executions, as long as the rest of
the conditions (PyTorch version, hardware, operation order, etc.) remain constant. In
this way, the creation of random tensors, the initialization of model parameters, and
other stochastic processes associated with PyTorch become deterministic.

The use of a fixed seed is especially important in experimentation and teaching
environments. In a teaching context, it allows all students to obtain the same results
when executing example notebooks, facilitating the follow-up of explanations and the
detection of possible conceptual or implementation errors. In a research and development
context, fixing the seed favors rigorous comparison between different models or
configurations, as it reduces variability attributable solely to chance.

It should be noted that, to achieve more complete reproducibility, it is often necessary
to also fix the seeds of other random number generators used in the same environment,
such as those from Python's standard libraries or NumPy.

#### Comprehensive Reproducibility Setup

For complete reproducibility across all components of a PyTorch project, it is
recommended to set seeds for all relevant libraries and configure deterministic behavior:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set seeds for reproducibility across all random number generators.

    Args:
        seed (int): The seed value to use for all RNGs
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy random number generator
    np.random.seed(seed)

    # PyTorch random number generators
    torch.manual_seed(seed)

    # PyTorch CUDA random number generator (for GPU operations)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure PyTorch to use deterministic algorithms
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For complete reproducibility in PyTorch >= 1.8
    # torch.use_deterministic_algorithms(True)  # Uncomment if needed

# Apply reproducibility settings
set_seed(42)

print("Reproducibility settings applied")
print(f"Random seed set to: 42")
```

#### Verifying Reproducibility

```python
# Test reproducibility with random tensor generation
def test_reproducibility():
    """Verify that setting the seed produces identical results"""

    # First run
    set_seed(42)
    tensor1 = torch.rand(3, 3)
    print("First run:")
    print(tensor1)

    # Second run with same seed
    set_seed(42)
    tensor2 = torch.rand(3, 3)
    print("\nSecond run:")
    print(tensor2)

    # Verify they are identical
    print(f"\nTensors are identical: {torch.equal(tensor1, tensor2)}")

test_reproducibility()
```
