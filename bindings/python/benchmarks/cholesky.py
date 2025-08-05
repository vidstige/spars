from pathlib import Path
from typing import List, Tuple
import time

import numpy as np


def generate_dense_spd_matrix(n, seed=42):
    np.random.seed(seed)
    A = np.random.randn(n, n)
    A = A @ A.T  # Symmetric positive semi-definite
    A += np.eye(n) * n  # Make positive definite
    b = np.random.rand(n)
    return A, b


def average(values: List[float]) -> float:
    assert values
    return sum(values) / len(values)


def load_equation_system(directory: Path, n: int) -> Tuple[np.ndarray, np.ndarray]:
    path = directory / f"spd_{n}.npz"
    with np.load(path) as data:
        return data["A"], data["b"]


def benchmark_spars(A_dense, b, repeats: int):
    from spars import CSC, cholesky, solve_cholesky
    A = CSC.from_dense(A_dense)
    factor_durations, solve_durations = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        L = cholesky(A)
        L.sort_indices()  # sort within each column
        factor_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        x = solve_cholesky(L, b)
        solve_durations.append(time.perf_counter() - start)
    return average(factor_durations), average(solve_durations)


def benchmark_scikit(A_dense, b, repeats):
    from scipy.sparse import csc_matrix  # type: ignore
    from sksparse.cholmod import cholesky  # type: ignore
    A = csc_matrix(A_dense)
    factor_durations, solve_durations = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        factor = cholesky(A)
        factor_durations.append(time.perf_counter() - start)

        start = time.perf_counter()
        x = factor(b)
        solve_durations.append(time.perf_counter() - start)

    return average(factor_durations), average(solve_durations)


if __name__ == "__main__":
    sizes = [100, 500, 1000, 2000]
    repeats = 3

    directory = Path("data")

    print("size,load,scikit-factor,scikit-solve,spars-factor,spars-solve,factor_fraction,solve_fraction")
    for n in sizes:
        start = time.perf_counter()
        A_dense, b = load_equation_system(directory, n)
        load_time = time.perf_counter() - start
        scikit_factor, scikit_solve = benchmark_scikit(A_dense, b, repeats)
        spars_factor, spars_solve = benchmark_spars(A_dense, b, repeats)
        print(f"{n},{load_time:.6f},{scikit_factor:.6f},{scikit_solve:.6f},{spars_factor:.6f},{spars_solve:.6f},{spars_factor/scikit_factor:.2f},{spars_solve/scikit_solve:.2f}")
