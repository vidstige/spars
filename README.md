![logo64x64.png](logo64x64.png)

# Spars - Sparse math without fortran
Lightweight sparse matrix c-library with python wrapper.

Focused on correctness, but should be fairly fast for many problems.

Already contains the following
* CSR & CSC sparse types
* LIL sparse type for each array creation
* Working sparse Cholesky factorization
* Solve function (Only for Hermitian positive-definite matrices)

Should be fairly fast as well, but hasn't been fully profiled and optimized.

## Developing
Building C only: `make`

To build & test the python bindings
* Use a virtual environment: `python3.11 -m venv venv`
* Activate: `source venv/bin/activate`
* Install bindings: `pip install -e bindings/python`
* If the above does not work, try this: `cd bindings/python/ && python setup.py build_ext --inplace`
* Run tests: `pytest bindings/python`
* Run benchmarks: `docker build --file benchmark.dockerfile . --tag spars-benchmarks && time docker run --rm -it spars-benchmarks`

### Profiling
On linux
1. Enable perf events: `sudo sysctl kernel.perf_event_paranoid=1`
2. Run with perf: `perf record --call-graph dwarf python bindings/python/benchmarks/cholesky.py`
3. Show report: `perf report`

## Alternatives
* [pysparse](https://github.com/PythonOptimizers/pysparse) - Created in the early 2000s and the last update was back in 2015. Not possible to install on modern python anymore
* [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) - Rather large library with everything from Fourier transforms, image processing to sparse matrices.

## Author
Samuel Carlsson <samuel.carlsson@gmail.com>