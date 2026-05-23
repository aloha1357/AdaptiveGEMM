import numpy as np
import pytest
import adaptive_gemm_py

def test_gpu_gemm_accuracy():
    np.random.seed(42)
    n = 256
    
    # Generate random double precision matrices
    A = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float64)
    B = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float64)

    # Reference computation using NumPy
    C_ref = A @ B

    # Computation using our custom AdaptiveGEMM GPU Engine
    engine = adaptive_gemm_py.AdaptiveGEMM()
    C_gpu = engine.gemm(A, B)

    # Verify maximum error is within expected tolerance
    np.testing.assert_allclose(C_gpu, C_ref, rtol=1e-5, atol=1e-8, err_msg="GPU GEMM result differs from NumPy reference")

if __name__ == "__main__":
    pytest.main([__file__])
