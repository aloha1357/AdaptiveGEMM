#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Simple CPU reference GEMM: C = A * B
// A: (m x k), B: (k x n)
static py::array_t<double> gemm_ref(py::array_t<double> a, py::array_t<double> b) {
    auto A = a.unchecked<2>();
    auto B = b.unchecked<2>();
    ssize_t m = A.shape(0);
    ssize_t k = A.shape(1);
    ssize_t kb = B.shape(0);
    ssize_t n = B.shape(1);
    if (k != kb) throw std::runtime_error("Inner dimensions must match");
    py::array_t<double> c({m, n});
    auto C = c.mutable_unchecked<2>();
    for (ssize_t i = 0; i < m; ++i) {
        for (ssize_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (ssize_t t = 0; t < k; ++t) sum += A(i,t) * B(t,j);
            C(i,j) = sum;
        }
    }
    return c;
}

PYBIND11_MODULE(adaptive_gemm_py, m) {
    m.doc() = "AdaptiveGEMM Python bindings (simple CPU reference GEMM)";
    m.def("gemm_ref", &gemm_ref, "Compute reference GEMM on CPU", py::arg("A"), py::arg("B"));
}
