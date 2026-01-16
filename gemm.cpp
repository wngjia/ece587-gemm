// gemm.cpp
// GEMM baseline: ijk vs ikj, row-major.
// Each kernel zeros C internally.
// Measures min / avg / max time across 3 runs.
//
// Usage:
//   ./gemm N
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 gemm.cpp -o gemm

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
 * ijk order
 */
void gemm_ijk(const float* A, const float* B, float* C, std::size_t N) {
    std::fill(C, C + N * N, 0.0f);

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t baseA = i * N;
        const std::size_t baseC = i * N;
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < N; ++k) {
                sum += A[baseA + k] * B[k * N + j];
            }
            C[baseC + j] = sum;
        }
    }
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    std::size_t N = std::stoull(argv[1]);
    if (N%256 != 0) {
        std::cerr << "Error: N must be a multiple of 256\n";
        return 1;
    }
    if (N > 8192) {
        std::cerr << "Error: N must be <= 8192\n";
        return 1;
    }

    std::size_t total = N * N;
    std::cout << "CWID A" << CWID
        << " " << time(0)
        << " N = " << N << ", "
        << (total*sizeof(float))/(1024.0*1024.0)
        << " MiB per matrix\n";

    std::vector<float> A(total), B(total), C(total), sums;
    fill_random(A);
    fill_random(B);

    if (N > 2048) {
        std::cout << "Skip gemm_ijk for N > 2048 as it will be too slow\n\n";
    } else {
        time_kernel("gemm_ijk", [&] {
            gemm_ijk(A.data(), B.data(), C.data(), N);
        }, C.data(), N, sums);
    }

    time_kernel("gemm_ikj", [&] {
        gemm_ikj(A.data(), B.data(), C.data(), N);
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
