// gemm_cuda.cu
// GEMM CUDA sample: tiled vs tiled 2x2.
// Measures min / avg / max time across 3 runs.
//
// Usage:
//   ./gemm_cuda N M
//
// Build:
//   nvcc -O2 gemm_cuda.cu -o gemm_cuda

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

constexpr int CWID = 12345678;

std::mt19937 g_rng(CWID);
std::uniform_real_distribution<float> g_dist(-1.0f, 1.0f);

void fill_random(std::vector<float>& x) {
    for (auto& v : x) v = g_dist(g_rng);
}

float sum_elements(const float* x, std::size_t NN) {
    return std::accumulate(x, x+NN, 0.0f);
}

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)


__global__ void gemm_tiled(const float* A, const float* B, float* C, int N) {
    extern __shared__ float shared[];
    const int M = static_cast<int>(blockDim.x);

    float* tile_A = shared;
    float* tile_B = shared+M*M;

    const int ti = static_cast<int>(blockIdx.y);
    const int tj = static_cast<int>(blockIdx.x);
    const int ii = static_cast<int>(threadIdx.y);
    const int jj = static_cast<int>(threadIdx.x);
    const int row = ti*M+ii;
    const int col = tj*M+jj;

    float sum = 0.0f;
    const int tiles = (N+M-1)/M;

    for (int tk = 0; tk < tiles; ++tk) {
        const int col_A = tk*M+jj;
        tile_A[ii*M+jj] =
            (row < N && col_A < N) ? A[row*N+col_A] : 0.0f;

        const int row_B = tk*M+ii;
        tile_B[ii*M+jj] =
            (row_B < N && col < N) ? B[row_B*N+col] : 0.0f;

        __syncthreads();

        for (int kk = 0; kk < M; ++kk) {
            sum += tile_A[ii*M+kk]*tile_B[kk*M+jj];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row*N+col] = sum;
    }
}

__global__ void gemm_tiled_2x2(const float* A, const float* B, float* C, int N) {
    extern __shared__ float shared[];
    const int M = static_cast<int>(blockDim.x);
    const int M2 = 2*M;

    float* tile_A = shared;
    float* tile_B = shared+M2*M2;

    const int ti = static_cast<int>(blockIdx.y);
    const int tj = static_cast<int>(blockIdx.x);
    const int ii = static_cast<int>(threadIdx.y);
    const int jj = static_cast<int>(threadIdx.x);

    const int row0 = ti*M2+ii;
    const int row1 = row0+M;
    const int col0 = tj*M2+jj;
    const int col1 = col0+M;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    const int tiles = (N+M2-1)/M2;

    // TODO: complete the loop below to update sum00/01/10/11
    // for (int tk = 0; tk < tiles; ++tk) {
    //    const int col0_A = tk*M2+jj;
    //    const int col1_A = col0_A+M;
    //    const int row0_B = tk*M2+ii;
    //    const int row1_B = row0_B+M;
    // }

    if (row0 < N && col0 < N) {
        C[row0*N+col0] = sum00;
    }
    if (row0 < N && col1 < N) {
        C[row0*N+col1] = sum01;
    }
    if (row1 < N && col0 < N) {
        C[row1*N+col0] = sum10;
    }
    if (row1 < N && col1 < N) {
        C[row1*N+col1] = sum11;
    }
}

template <typename LaunchFn>
void time_kernel(const char* name, LaunchFn&& launch, float* d_C,
    std::size_t N, std::vector<float>& sums) {
    constexpr int RUNS = 3;
    double min_ms = std::numeric_limits<double>::infinity();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    std::vector<float> C(N*N);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int r = 0; r < RUNS; ++r) {
        CHECK_CUDA(cudaMemset(d_C, 0, N*N*sizeof(float)));
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        CHECK_CUDA(cudaMemcpy(C.data(), d_C, N*N*sizeof(float),
            cudaMemcpyDeviceToHost));

        sums.push_back(sum_elements(C.data(), N*N));
        min_ms = std::min(min_ms, static_cast<double>(ms));
        max_ms = std::max(max_ms, static_cast<double>(ms));
        sum_ms += ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " " << name << " : min " << min_ms
        << " ms, avg " << (sum_ms/RUNS)
        << " ms, max " << max_ms << " ms\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N M\n";
        return 1;
    }

    const int N = std::atoi(argv[1]);
    const int M = std::atoi(argv[2]);
    if (N <= 0) {
        std::cerr << "Error: N must be > 0\n";
        return 1;
    }
    if (M <= 0) {
        std::cerr << "Error: M must be > 0\n";
        return 1;
    }

    if (N%256 != 0) {
        std::cerr << "Error: N must be a multiple of 256\n";
        return 1;
    }
    if (N > 16384) {
        std::cerr << "Error: N must be <= 16384\n";
        return 1;
    }
    if ((M&(M-1)) != 0) {
        std::cerr << "Error: M must be power of 2\n";
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(0));

    const std::size_t total = N*N;
    const std::size_t bytes = total*sizeof(float);
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " N = " << N << ", M = " << M << ", "
        << bytes/(1024.0*1024.0)
        << " MiB per matrix\n";

    std::vector<float> A(total), B(total), sums;
    fill_random(A);
    fill_random(B);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    time_kernel("gemm_tiled", [&] {
        const dim3 dim_threads(M, M);
        const dim3 dim_blocks((N+M-1)/M, (N+M-1)/M);
        const std::size_t shared_bytes = 2ULL*M*M*sizeof(float);
        gemm_tiled<<<dim_blocks, dim_threads, shared_bytes>>>(
            d_A, d_B, d_C, N);
    }, d_C, N, sums);

    time_kernel("gemm_tiled_2x2", [&] {
        const dim3 dim_threads(M, M);
        const dim3 dim_blocks((N+M*2-1)/(M*2), (N+(M*2)-1)/(M*2));
        const std::size_t shared_bytes = 8ULL*M*M*sizeof(float);
        gemm_tiled_2x2<<<dim_blocks, dim_threads, shared_bytes>>>(
            d_A, d_B, d_C, N);
    }, d_C, N, sums);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    const auto minmax = std::minmax_element(sums.begin(), sums.end());
    const bool passed = std::fabs(*minmax.second-*minmax.first)
        < std::fabs(*minmax.second+*minmax.first)*1e-3;
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " sums: min " << *minmax.first
        << ", max " << *minmax.second
        << (passed ? ", passed" : ", failed")
        << "\n";

    return 0;
}
