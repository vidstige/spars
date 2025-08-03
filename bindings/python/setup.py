# Warning: this file is still needed for the C extension to compile correctly.
import platform
from setuptools import setup, Extension, find_packages

import numpy


# Figure out suitable compiler flags for the current platform.
def detect_machine_flag():
    machine = platform.machine().lower()
    processor = platform.processor().lower()
    if platform.system() == "Darwin":
        if "arm" in machine or "apple" in processor or "m1" in processor:
            return "-mcpu=apple-m1"
    return "-march=native"


setup(
    name="sparsely",
    packages=find_packages(),
    package_data={"sparsely": ["py.typed"]},
    install_requires=["numpy"],
    ext_modules=[
        Extension(
            "sparsely._sparse_c",
            sources=[
                "sparsely/sparsely_module.c",
                "sparsely/sparsely_csr.c",
                "sparsely/sparsely_csc.c",
                "sparsely/sparsely_add.c",
                "sparsely/sparsely_mul.c",
                "sparsely/sparsely_cholesky.c",
                "../../src/csr.c",
                "../../src/csc.c",
                "../../src/cholesky.c",
                "../../src/cholesky_solve.c",
                "../../src/dense.c",
                "../../src/add.c",
                "../../src/mul.c",
                "../../src/alloc.c",
                "../../src/blasphemy.c",
            ],
            include_dirs=["../../include", numpy.get_include()],
            extra_compile_args=[
                "-O3",
                detect_machine_flag(),
                "-ffast-math",
                "-flto",
                "-funroll-loops",
                "-ftree-vectorize",
            ],
            extra_link_args=["-flto"],
        )
    ],
)