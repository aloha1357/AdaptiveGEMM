#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <string>

#include "AdaptiveOzaki.h"

namespace py = pybind11;
using namespace ozaki;

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err));  \
        }                                                                                     \
    } while (0)

struct CudaDeleter {
    void operator()(double* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

using unique_cuda_ptr = std::unique_ptr<double, CudaDeleter>;

// Simple CPU reference GEMM: C = A * B
// A: (m x k), B: (k x n)
static py::array_t<double> gemm_ref(py::array_t<double> a, py::array_t<double> b) {
    auto A = a.unchecked<2>();
    auto B = b.unchecked<2>();
    py::ssize_t m = A.shape(0);
    py::ssize_t k = A.shape(1);
    py::ssize_t kb = B.shape(0);
    py::ssize_t n = B.shape(1);
    if (k != kb) throw std::runtime_error("Inner dimensions must match");
    py::array_t<double> c({m, n});
    auto C = c.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < m; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (py::ssize_t t = 0; t < k; ++t) sum += A(i,t) * B(t,j);
            C(i,j) = sum;
        }
    }
    return c;
}

class AdaptiveGEMMWrapper {
public:
    AdaptiveGEMMWrapper() {
        // Default configuration
        config_.mode = ExecutionMode::Phase22; 
    }

    py::array_t<double> gemm(py::array_t<double> a, py::array_t<double> b) {
        py::buffer_info a_info = a.request();
        py::buffer_info b_info = b.request();

        if (a_info.ndim != 2 || b_info.ndim != 2) {
            throw std::runtime_error("Inputs must be 2D numpy arrays");
        }

        int m = static_cast<int>(a_info.shape[0]);
        int k = static_cast<int>(a_info.shape[1]);
        int kb = static_cast<int>(b_info.shape[0]);
        int n = static_cast<int>(b_info.shape[1]);

        if (k != kb) {
            throw std::runtime_error("Inner dimensions must match");
        }

        // Ensure contiguous arrays
        a = py::array_t<double, py::array::c_style | py::array::forcecast>(a);
        b = py::array_t<double, py::array::c_style | py::array::forcecast>(b);

        const double* h_A = static_cast<const double*>(a.data());
        const double* h_B = static_cast<const double*>(b.data());

        py::array_t<double, py::array::c_style> c({m, n});
        double* h_C = static_cast<double*>(c.mutable_data());

        double* d_A_raw = nullptr;
        double* d_B_raw = nullptr;
        double* d_C_raw = nullptr;

        size_t size_A = m * k * sizeof(double);
        size_t size_B = k * n * sizeof(double);
        size_t size_C = m * n * sizeof(double);

        CHECK_CUDA(cudaMalloc(&d_A_raw, size_A));
        unique_cuda_ptr d_A(d_A_raw);

        CHECK_CUDA(cudaMalloc(&d_B_raw, size_B));
        unique_cuda_ptr d_B(d_B_raw);

        CHECK_CUDA(cudaMalloc(&d_C_raw, size_C));
        unique_cuda_ptr d_C(d_C_raw);

        CHECK_CUDA(cudaMemcpy(d_A.get(), h_A, size_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B.get(), h_B, size_B, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C.get(), 0, size_C));

        AdaptiveOzakiEngine engine(config_);
        engine.execute(d_A.get(), d_B.get(), d_C.get(), m, n, k);

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C.get(), size_C, cudaMemcpyDeviceToHost));

        return c;
    }

private:
    OzakiConfig config_;
};

PYBIND11_MODULE(adaptive_gemm_py, m) {
    m.doc() = "AdaptiveGEMM Python bindings (GPU-accelerated numpy support)";
    m.def("gemm_ref", &gemm_ref, "Compute reference GEMM on CPU", py::arg("A"), py::arg("B"));

    py::class_<AdaptiveGEMMWrapper>(m, "AdaptiveGEMM")
        .def(py::init<>())
        .def("gemm", &AdaptiveGEMMWrapper::gemm, "Compute GEMM on GPU using AdaptiveOzakiEngine", py::arg("A"), py::arg("B"));
}
