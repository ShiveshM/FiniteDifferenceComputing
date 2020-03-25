# Finite Difference Computing

![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)

Notes from Langtangen's "Finite Difference Computing with Exponential Decay Models", [published by Springer](http://link.springer.com/10.1007/978-3-319-29439-1).

### Chapters
1. **Algorithms and implementations**
    * The Forward Euler scheme

### Dependencies
* [`Python`](https://www.python.org/) >= 3.7
* [`NumPy`](http://www.numpy.org/)

### Usage
Overview of chapters is given above. To run all code for a particular chapter,
e.g. chapter 1, simply run
```
python chap1.py
```
To specify a particular method(s), then enter the name of the method after the
filename, e.g. to run the methods `forward_euler` and `backward_euler` inside
`chap1.py`, do
```
python chap1.py forward_euler backward_euler
```
