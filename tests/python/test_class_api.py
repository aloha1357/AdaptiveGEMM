import numpy as np
from adaptive_gemm_py import AdaptiveGEMM

def test_gemm_class_small():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    engine = AdaptiveGEMM()
    C = engine.gemm(A, B)
    expected = A.dot(B)
    assert np.allclose(C, expected)
