import numpy as np
from spars import CSR

from .matrices import easy3x3

def test_csr_add_csr():
    lhs_dense, rhs_dense = easy3x3(), easy3x3()
    a = CSR.from_dense(lhs_dense)
    b = CSR.from_dense(rhs_dense)
    c = a + b
    expected = lhs_dense + rhs_dense
    np.testing.assert_allclose(c.todense(), expected)
