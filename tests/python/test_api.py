import numpy as np
from adaptive_gemm_py import gemm_ref

def test_gemm_small():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    C = gemm_ref(A, B)
    expected = A.dot(B)
    assert np.allclose(C, expected)
