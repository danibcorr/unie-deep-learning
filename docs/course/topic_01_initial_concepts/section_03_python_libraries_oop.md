# Libraries and Software Design

The development of libraries in Python is based on design principles that seek to
maximize clarity, reusability, and long-term code maintenance. In this context,
**modularization** and **object-oriented programming (OOP)** constitute fundamental
pillars, especially in Deep Learning libraries such as TensorFlow or PyTorch, whose
design reflects engineering decisions designed to scale to complex and collaborative
projects. The ultimate goal is to adopt the mindset of a professional Machine Learning
engineer, capable of building reusable, extensible code that can be easily integrated
into other projects.

Modularization in Python consists of dividing code into well-defined logical units,
avoiding monolithic files and favoring separation of responsibilities. This approach
leads to cleaner and more maintainable code, in which each module fulfills a specific
function within the system. When this modularization is combined with object-oriented
design, the result is a coherent, structured library aligned with industry practices.

A key step in this process is converting the project into an **installable package**, so
that it can be distributed and installed in other projects using modern tools like
**uv**. This packaging capability reinforces code reusability and allows validation that
the design is truly independent of the environment in which it is developed.

## Object-Oriented Programming

Object-Oriented Programming is a paradigm that organizes software around objects, rather
than focusing solely on functions and control flows. Each object encapsulates data,
represented by attributes, and behavior, defined through methods that operate on that
data. This model facilitates modularity, reusability, and code scalability—essential
qualities in complex systems like Deep Learning libraries.

In Python, OOP offers a remarkable balance between expressiveness and simplicity. Through
the use of classes, inheritance, and polymorphism, it is possible to build flexible
structures that adapt to new needs without introducing drastic changes to existing code.
This approach favors maintainability and reduces the likelihood of errors when extending
or modifying functionalities.

### Classes and Objects

A **class** acts as a mold or template from which objects are created, which are concrete
instances of that class. Each object has attributes that describe its state and methods
that define its behavior. In Python, classes are defined using the `class` keyword, and
the special method `__init__` is used as a constructor to initialize the object's
attributes at the time of creation.

```python
class ClassName:

    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    def some_method(self):
        print("This is a method inside the class")
```

In this example, `self` refers to the concrete instance of the object and allows access
to its attributes and methods. The explicit use of `self` is an essential feature of
Python's object model.

A practical case illustrates this concept more concretely:

```python
class Car:

    def __init__(self, brand, model, upgraded, car_access):
        self.brand = brand
        self.model = model
        self.upgraded = upgraded
        self.car_access = car_access

my_car = Car("Toyota", "Corolla", True, ["Juan", "Maria"])
print(f"My car is a {my_car.brand} {my_car.model}")
```

Here the `Car` class is defined with several attributes that describe the object's state.
The `my_car` instance represents a specific car, with its own values for each attribute,
which exemplifies how a class can generate multiple independent objects.

### Methods and Attributes

**Attributes** represent the characteristics of an object, while **methods** describe the
actions it can perform. In Python, instance attributes, which are specific to each
object, are distinguished from class attributes, which are shared by all instances of the
same class.

```python
class Dog:

    species = "mammal"

    def __init__(self, breed, name, age):
        self.breed = breed
        self.name = name
        self.age = age

    def sound(self):
        return "Woof!"

    def information(self):
        print(
            f"Name: {self.name}, "
            f"Breed: {self.breed}, "
            f"Age: {self.age}, "
            f"Species: {self.species}"
        )

if __name__ == "__main__":
    my_dog = Dog("Labrador", "Fido", 3)
    my_dog.information()
```

In this example, `species` is a class attribute common to all dogs, while `breed`,
`name`, and `age` are instance attributes. This distinction is especially useful for
modeling shared concepts versus individual characteristics.

### Inheritance and Polymorphism

**Inheritance** allows creating new classes from existing ones, promoting code
reusability and progressive specialization. A subclass inherits the attributes and
methods of its base class but can extend or redefine them as needed.

```python
class Animal:

    def __init__(self, name):
        self.name = name

    def who_am_i(self):
        print("I am an animal")

    def eat(self):
        print("I am eating")

class Dog(Animal):

    def who_am_i(self):
        print(f"I am a dog named {self.name}")

my_dog = Dog("Fido")
my_dog.who_am_i()
my_dog.eat()
```

In this case, `Dog` inherits from `Animal`, reusing the `eat` method and redefining the
`who_am_i` method. This mechanism reduces code duplication and clarifies hierarchical
relationships between concepts.

**Polymorphism** complements inheritance by allowing different objects to respond to the
same method in different ways. This translates into common interfaces with specific
behaviors depending on the concrete class of the object.

```python
class Dog:

    def __init__(self, name):
        self.name = name

    def sound(self):
        print(f"The dog {self.name} barks")

class Cat:

    def __init__(self, name):
        self.name = name

    def sound(self):
        print(f"The cat {self.name} meows")

my_dog = Dog("Fido")
my_cat = Cat("Meow")

my_dog.sound()
my_cat.sound()
```

Both classes implement the `sound` method, but each does so differently, allowing objects
to be treated uniformly without losing specificity.

### Abstract Classes

**Abstract classes** define a common interface that subclasses must implement, without
necessarily providing a complete implementation. Although Python does not enforce strict
abstraction by default, it is possible to simulate it through methods that raise
exceptions when not implemented.

```python
class Animal:

    def __init__(self, name):
        self.name = name

    def sound(self):
        raise NotImplementedError("Subclass must implement this method")

class Dog(Animal):

    def sound(self):
        return f"{self.name} makes woof!"

my_dog = Dog("Fido")
print(my_dog.sound())
```

This pattern forces subclasses to define certain behaviors, ensuring consistency in the
class hierarchy design.

## Modules and Packages

Organizing code into **modules** and **packages** is an essential step for building
reusable libraries. A module is simply a `.py` file that contains definitions of
functions, classes, or variables. A package is a directory that groups several related
modules and includes an `__init__.py` file, necessary for Python to recognize it as such.

### Importing Modules and Using Libraries in Python

Python allows importing custom or external modules to reuse already implemented
functionality. Traditionally, the installation of external libraries has been done using
**pip**, which downloads packages from **PyPI**, Python's official repository. For
example, the installation and use of the `colorama` library to print colored text is done
as follows:

```bash
pip install colorama
```

```python
from colorama import init, Fore

init()
print(Fore.RED + "Test text")
```

### Package Structure and Code Reusability

A clear modular structure facilitates code readability and its reuse in different
projects. An example of organizing a project with packages and subpackages is as follows:

```python
# main.py
from package78 import some_main_script as p
from package78.Subpackages import mysubscript as s

p.main_report()
s.sub_report()
```

```python
# package78/some_main_script.py
def main_report():
    print("Hello, I am a function inside my main script.")
```

```python
# package78/Subpackages/mysubscript.py
def sub_report():
    print("Hello, I am a function inside my subscript.")
```

This structure reflects how Python locates and organizes code, and constitutes the basis
for converting a project into an installable package. By packaging the project and
distributing it using modern tools like **uv**, a professional workflow is consolidated
that prioritizes reproducibility, reusability, and software design quality.
