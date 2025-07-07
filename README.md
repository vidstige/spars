![logo64x64.png](logo64x64.png)

# Sparsely
Lightweight sparse matrix c-library with python wrapper.

Focused on correctness, but should be fairly fast for many problems.

## Developing
Building C only: `make`

To build & test the python bindings
* Use a virtual environment: `python3.11 -m venv venv`
* Activate: `source venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`
* Install bindings: `pip install -e bindings/python`
* Run tests: `pytest bindings/python`

## Alternatives
* [pysparse](https://github.com/PythonOptimizers/pysparse) - Created in the early 2000s and the last update was back in 2015. Not possible to install on modern python anymore
* [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) - Rather large library with everything from Fourier transforms, image processing to sparse matrices.

## Author
Samuel Carlsson <samuel.carlsson@gmail.com>