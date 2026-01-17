# Data Structures

In Python, data structures constitute an essential element for storing, organizing, and
manipulating information efficiently. In the context of Deep Learning and scientific
computing, a solid understanding of these structures is key, as they serve as the
conceptual foundation upon which more complex abstractions are built, such as tensors,
datasets, and batches. Below are the most common data structures in Python, explained
progressively and with a practical approach.

## Lists

**Lists** are data structures that allow storing ordered and mutable sequences of
elements. Unlike other languages, lists in Python can contain elements of different types
and their size is dynamic, meaning they can grow or shrink during program execution.
Indexing starts at zero, and negative indices allow accessing elements starting from the
end of the list.

A list is defined using brackets and separating elements with commas:

```python
friends_list = ["Jorge", "Fran", "Ricardo"]
```

It is also possible to initialize an empty list, which is useful when you want to build
it progressively:

```python
lista = []
```

Access to elements is done through indices, and Python offers flexible syntax for
selecting individual elements or complete subsets:

```python
friends_list = ["Jorge", "Fran", "Ricardo"]

print(f"The first friend is {friends_list[0]}")
print(f"My hometown friend is {friends_list[-1]}")
print(friends_list[0:2])
print(friends_list)
```

This indexing and slicing capability makes lists especially suitable for representing
data sequences, such as sets of examples or intermediate calculation results.

### Main List Methods

Lists have a wide set of methods that allow modifying their content and structure. Among
the most relevant are the following:

| Function                | Definition                                                          |
| ----------------------- | ------------------------------------------------------------------- |
| `list[index] = x`       | Changes the element at the specified index to `x`.                  |
| `list.extend(x)`        | Adds the elements of `x` to the end of the list.                    |
| `list.append(x)`        | Adds a single element `x` to the end of the list.                   |
| `list.insert(index, x)` | Inserts `x` at the indicated position.                              |
| `list.remove(x)`        | Removes the first occurrence of `x`.                                |
| `list.clear()`          | Completely empties the list.                                        |
| `list.pop()`            | Removes and returns the last element or the one indicated by index. |
| `list.index(x)`         | Returns the index of the first occurrence of `x`.                   |
| `list.count(x)`         | Counts how many times `x` appears.                                  |
| `list.sort()`           | Sorts the list in ascending order.                                  |
| `list.reverse()`        | Reverses the order of elements.                                     |
| `list2 = list1.copy()`  | Creates a shallow copy of the list.                                 |
| `max(list)`             | Returns the maximum value.                                          |
| `min(list)`             | Returns the minimum value.                                          |
| `del list[x]`           | Removes the element at index `x`.                                   |

### `for` Loops and List Comprehension

Iteration over lists using `for` loops is a common operation in Python. Additionally, the
language offers **list comprehension**, a concise and expressive syntax for generating
new lists from existing sequences.

```python
my_list = [letter for letter in "Hello"]
print(my_list)

my_list = [number ** 2 for number in range(0, 20, 2)]
print(my_list)

celsius = [0, 10, 20, 34.5]
fahrenheit = [((9/5) * temp + 32) for temp in celsius]
print(fahrenheit)

my_list = [number ** 2 for number in range(0, 15, 2) if number % 2 == 0]
print(my_list)
```

This mechanism is especially useful in data science and machine learning, where it is
common to apply repetitive transformations over collections of values.

### Nested Lists and Matrix Representation

Lists can contain other lists, which allows representing multidimensional structures like
matrices or data tables:

```python
number_grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0]
]

print(number_grid[2][2])
```

Although in real projects matrices are usually managed with libraries like NumPy or
PyTorch, this model helps understand the underlying logic of multidimensional data.

## Tuples

**Tuples** are ordered and immutable sequences. Once created, their elements cannot be
modified, which makes them a suitable option when you want to guarantee data integrity.
Additionally, they are usually more memory-efficient and slightly faster than lists.

```python
coordinates = (4, 5)

print(f"Complete coordinate {coordinates}")
print(f"First coordinate {coordinates[0]} and second coordinate {coordinates[1]}")
```

Tuples are frequently used to group related values, for example, coordinates, parameters,
or multiple results from a function.

### Tuple Methods

Although they are immutable, tuples offer some basic methods:

| Function         | Description                                       |
| ---------------- | ------------------------------------------------- |
| `tuple.count(x)` | Returns the number of times `x` appears.          |
| `tuple.index(x)` | Returns the index of the first occurrence of `x`. |

## Sets

**Sets** are unordered collections of unique elements. They do not allow duplicates,
which makes them especially useful for removing repetitions or for performing set theory
operations like unions and intersections.

```python
my_set = set()
my_set.add(1)
my_set.add(1)

my_new_set = {'a', 'b', 'c'}
```

### Set Methods

Sets provide efficient operations for comparing collections:

| Function                   | Definition                                         |
| -------------------------- | -------------------------------------------------- |
| `s.add(x)`                 | Adds an element to the set.                        |
| `s.clear()`                | Removes all elements.                              |
| `sc = s.copy()`            | Creates a copy of the set.                         |
| `s1.difference(s2)`        | Returns the elements of `s1` that are not in `s2`. |
| `s1.difference_update(s2)` | Removes from `s1` the elements present in `s2`.    |
| `s.discard(elem)`          | Removes `elem` without error if it doesn't exist.  |
| `s1.intersection(s2)`      | Returns the common elements.                       |
| `s1.issubset(s2)`          | Checks if `s1` is a subset of `s2`.                |
| `s1.union(s2)`             | Returns the union of both sets.                    |

## Dictionaries

**Dictionaries** store information in **key-value** pairs, where each key is unique. They
are mutable and allow very efficient data access, making them one of the most used
structures in Python.

```python
month_conversion = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March"
}

print(month_conversion["Jan"])
print(month_conversion.get("Jan"))

key = "Daniel"
print(month_conversion.get(key, f"The key {key} is not in the dictionary"))
```

### Main Dictionary Methods

| Function              | Definition                   |
| --------------------- | ---------------------------- |
| `dictionary.items()`  | Returns the key-value pairs. |
| `dictionary.keys()`   | Returns the keys.            |
| `dictionary.values()` | Returns the values.          |

### Practical Cases and Compound Structures

Dictionaries can be nested or combined with lists to represent complex structures:

```python
dictionary = {"k3": {'insideKey': 100}}
print(dictionary["k3"]['insideKey'])
```

Iteration over dictionaries allows traversing keys, values, or both simultaneously:

```python
d = {'k1': 1, 'k2': 2}

for key in d.keys():
    print(key)

for value in d.values():
    print(value)

for item in d.items():
    print(item)
```

A more realistic example combines lists and dictionaries to model entities with variable
attributes:

```python
clients = [
    {"name": "Daniel", "animals": ["Pakito", "Pakon", "Pakonazo"]},
    {"name": "Clemencia", "animals": ["Rodolfo"]},
    {"name": "Carolina"}
]

for client in clients:
    print(f"{client['name']} has: {client.get('animals', 'No animals')}")
```

## Tensors

In the field of Deep Learning, **tensors** constitute the fundamental data structure upon
which machine learning models and algorithms are built. Conceptually, a tensor can be
understood as a generalization of the structures seen previously: a scalar is a
zero-order tensor, a vector is a one-dimensional tensor, a matrix is a two-dimensional
tensor and, more generally, a higher-order tensor represents data in multiple dimensions.
This abstraction allows describing complex information in a uniform way, such as images,
temporal sequences, or data batches.

From a practical point of view, tensors are not usually implemented using nested Python
lists, as these are not optimized for intensive numerical computing. Instead, specialized
libraries such as **NumPy**, **PyTorch**, or **TensorFlow** are used, which provide
efficient implementations based on contiguous memory, vectorized operations and, in many
cases, GPU acceleration.

A tensor is mainly characterized by its **shape**, which describes the number of
dimensions and the size of each one, and by its **data type** (_dtype_), which indicates
how values are stored in memory. These properties are essential to guarantee
compatibility between operations and to optimize computational performance.

In PyTorch, for example, tensor creation is done explicitly from existing data or through
initialization functions:

```python
import torch

# Scalar tensor
a = torch.tensor(3.14)

# One-dimensional tensor (vector)
v = torch.tensor([1, 2, 3])

# Two-dimensional tensor (matrix)
m = torch.tensor([[1, 2], [3, 4]])
```

Each of these tensors has a specific shape that can be queried directly, which
facilitates dimension verification before applying mathematical operations. This explicit
control of dimensions is crucial in Deep Learning, where shape mismatches are often a
common source of errors.

Operations on tensors are defined in a vectorized manner, which allows expressing complex
calculations concisely and efficiently. For example, addition or multiplication between
tensors is performed element-wise whenever their shapes are compatible, following
well-defined broadcasting rules. This computational model avoids explicit loops and
leverages internal system optimizations.

Another fundamental aspect of tensors in Deep Learning libraries is their integration
with **automatic differentiation** mechanisms. In PyTorch, tensors can be marked for
gradient tracking, which allows automatically calculating derivatives during the model
training process. This capability makes tensors not just data containers, but active
elements within the computational graph.

Finally, tensors facilitate code execution on both CPU and GPU without modifying the
program logic. By moving a tensor to a specific device, associated operations are
automatically executed on that hardware, which reinforces the abstraction and allows
focusing on model design rather than low-level details. Overall, tensors represent the
natural evolution of basic data structures toward an optimized and expressive
computational model, indispensable in the modern development of Deep Learning systems.
