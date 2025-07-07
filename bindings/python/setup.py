from setuptools import setup, Extension, find_packages
import numpy

# Warning: this file is still needed for the C extension to compile correctly.

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
                "sparsely/sparsely_cholesky.c",
                "../../src/csr.c",
                "../../src/cholesky.c",
                "../../src/cholesky_solve.c",
                "../../src/dense.c",
                "../../src/mul.c",
            ],
            include_dirs=["../../include", numpy.get_include()],
        )
    ],
)