// gemm_tiled.cpp
// GEMM optimized: ikj row-major vs tiled.
// Each kernel zeros C internally.
// Measures min / avg / max time across 3 runs.
//
// Usage:
//   OMP_NUM_THREADS=4 ./gemm_tiled N M
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 gemm_tiled.cpp -o gemm_tiled

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <omp.h>

using Clock = std::chrono::steady_clock;

constexpr int CWID = 12345678;

std::mt19937 g_rng(CWID);
std::uniform_real_distribution<float> g_dist(-1.0f, 1.0f);

void fill_random(std::vector<float>& x) {
    for (auto& v : x) v = g_dist(g_rng);
}

float sum(const float* x, std::size_t NN) {
    return std::accumulate(x, x + NN, 0.0f);
}

/*
 * ikj order
 */
void gemm_ikj(const float* A, const float* B, float* C, std::size_t N) {
    std::fill(C, C + N * N, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t baseA = i * N;
        const std::size_t baseC = i * N;
        for (std::size_t k = 0; k < N; ++k) {
            const float a = A[baseA + k];
            const std::size_t baseB = k * N;
            for (std::size_t j = 0; j < N; ++j) {
                C[baseC + j] += a * B[baseB + j];
            }
        }
    }
}

void copy_to_tiled(const float* src, float* dst,
    std::size_t N, std::size_t M) {
    const std::size_t tiles = N / M;
    const std::size_t M2 = M * M;
    for (std::size_t ti = 0; ti < tiles; ++ti) {
        const std::size_t i0 = ti * M;
        for (std::size_t tj = 0; tj < tiles; ++tj) {
            const std::size_t j0 = tj * M;
            const std::size_t base_tile = (ti * tiles + tj) * M2;
            for (std::size_t ii = 0; ii < M; ++ii) {
                const std::size_t base_src = (i0 + ii) * N + j0;
                const std::size_t base_dst = base_tile + ii * M;
                for (std::size_t jj = 0; jj < M; ++jj) {
                    dst[base_dst + jj] = src[base_src + jj];
                }
            }
        }
    }
}

void gemm_tiled(const float* At, const float* Bt, float* Ct,
    std::size_t N, std::size_t M) {
    std::fill(Ct, Ct + N * N, 0.0f);

    const std::size_t tiles = N / M;
    const std::size_t M2 = M * M;
    for (std::size_t ti = 0; ti < tiles; ++ti) {
        for (std::size_t tj = 0; tj < tiles; ++tj) {
            const std::size_t baseC_tile = (ti * tiles + tj) * M2;
            for (std::size_t tk = 0; tk < tiles; ++tk) {
                const std::size_t baseA_tile = (ti * tiles + tk) * M2;
                const std::size_t baseB_tile = (tk * tiles + tj) * M2;
                for (std::size_t ii = 0; ii < M; ++ii) {
                    const std::size_t baseCt = baseC_tile + ii * M;
                    const std::size_t baseAt = baseA_tile + ii * M;
                    for (std::size_t kk = 0; kk < M; ++kk) {
                        const float a = At[baseAt + kk];
                        const std::size_t baseBt = baseB_tile + kk * M;
                        for (std::size_t jj = 0; jj < M; ++jj) {
                            Ct[baseCt + jj] += a * Bt[baseBt + jj];
                        }
                    }
                }
            }
        }
    }
}

template <typename Fn>
void time_kernel(const char* name, Fn&& fn, float* C, std::size_t N,
    std::vector<float> &sums) {
    constexpr int RUNS = 3;
    double min_ms = std::numeric_limits<double>::infinity();
    double max_ms = 0.0;
    double sum_ms = 0.0;

    std::cout << std::fixed << std::setprecision(3);

    for (int r = 0; r < RUNS; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        // sum result to avoid dead-code elimination
        sums.push_back(sum(C, N * N));

        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " " << name << " : min " << min_ms
        << " ms, avg " << (sum_ms / RUNS)
        << " ms, max " << max_ms << " ms\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N M\n";
        return 1;
    }

    std::size_t N = std::stoull(argv[1]);
    std::size_t M = std::stoull(argv[2]);
    if (N%256 != 0) {
        std::cerr << "Error: N must be a multiple of 256\n";
        return 1;
    }
    if (N > 8192) {
        std::cerr << "Error: N must be <= 8192\n";
        return 1;
    }
    if (M == 0) {
        std::cerr << "Error: M must be > 0\n";
        return 1;
    }
    if (N % M != 0) {
        std::cerr << "Error: N must be a multiple of " << M << "\n";
        return 1;
    }

    std::size_t total = N * N;
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " N = " << N << ", M = " << M << ", "
        << (total*sizeof(float))/(1024.0*1024.0)
        << " MiB per matrix\n";
    std::cout << "OMP_NUM_THREADS = "
        << omp_get_max_threads() << "\n";

    std::vector<float> A(total), B(total), C(total), sums;
    fill_random(A);
    fill_random(B);
    
    std::vector<float> At(total), Bt(total);
    copy_to_tiled(A.data(), At.data(), N, M);
    copy_to_tiled(B.data(), Bt.data(), N, M);

    time_kernel("gemm_ikj", [&] {
        gemm_ikj(A.data(), B.data(), C.data(), N);
    }, C.data(), N, sums);

    time_kernel("gemm_tiled", [&] {
        gemm_tiled(At.data(), Bt.data(), C.data(), N, M);
    }, C.data(), N, sums);

    auto minmax = std::minmax_element(sums.begin(), sums.end());
    bool passed = std::fabs(*minmax.second-*minmax.first)
        < std::fabs(*minmax.second+*minmax.first)*1e-3;
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " sums: min " << *minmax.first
        << ", max " << *minmax.second
        << (passed? ", passed": ", failed")
        << "\n";

    return 0;
}
