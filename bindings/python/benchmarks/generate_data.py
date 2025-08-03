from pathlib import Path
import numpy as np


def random_semi_positive_definite(n: int) -> np.ndarray:
    A = np.random.randn(n, n)
    A = A @ A.T
    A += np.eye(n) * n
    return A


def random_vector(n: int) -> np.ndarray:
    return np.random.rand(n)


def main():
    seed = 42
    np.random.seed(seed)
    sizes = [100, 500, 1000, 2000]
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    for n in sizes:
        A, b = random_semi_positive_definite(n), random_vector(n)
        path = output_dir / f"spd_{n}.npz"
        np.savez_compressed(path, A=A, b=b)
        print(f"saved {path}")


if __name__ == "__main__":
   main()
