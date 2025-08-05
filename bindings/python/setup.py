# Warning: this file is still needed for the C extension to compile correctly.
import os
import platform
import subprocess
from setuptools import setup, Extension, find_packages

import numpy


# figure out package version 
def get_version():
    version = os.environ.get("GITHUB_REF_NAME")
    if version and version.startswith("v"):
        return version.lstrip("v")
    short_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
    ).decode().strip()
    return f"0.0.1+dev.{short_sha}"


# Figure out suitable compiler flags for the current platform.
def detect_machine_flag():
    machine = platform.machine().lower()
    processor = platform.processor().lower()
    if platform.system() == "Darwin":
        if "arm" in machine or "apple" in processor or "m1" in processor:
            return "-mcpu=apple-m1"
    return "-march=native"


extra_compile_args = [
    "-O3",
    detect_machine_flag(),
    "-ffast-math",
    "-fno-math-errno",
    "-flto",
    "-funroll-loops",
    "-ftree-vectorize",
]


extra_link_args = [
    "-flto",
]


setup(
    version=get_version(),
    packages=find_packages(),
    package_data={"spars": ["py.typed"]},
    install_requires=["numpy"],
    ext_modules=[
        Extension(
            "spars._sparse_c",
            sources=[
                "spars/spars_module.c",
                "spars/spars_csr.c",
                "spars/spars_csc.c",
                "spars/spars_add.c",
                "spars/spars_mul.c",
                "spars/spars_cholesky.c",
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
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
)