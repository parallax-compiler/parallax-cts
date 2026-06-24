// Parallax CTS — Differential Test Harness
// -----------------------------------------
// Correctness method: differential testing against a ground truth.
//
//   ground truth = std::execution::seq   (always runs on the CPU, serially)
//   candidate    = std::execution::par   (Parallax GPU offload when built with the
//                                          plugin; plain CPU otherwise)
//
// Every case runs the SAME algorithm twice on identical inputs and diffs the
// results, sweeping a range of sizes and value distributions. When built WITHOUT
// the Parallax plugin, both policies execute on the CPU, so the suite becomes a
// self-test of the harness (everything must pass). When built WITH the plugin,
// any divergence between seq and par is a real GPU-offload correctness bug.
//
// This header is intentionally dependency-free (only the C++ standard library) and
// pure ISO C++ so the exact same source compiles for both build modes.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace pcts {

// ---------------------------------------------------------------------------
// Element comparison
// ---------------------------------------------------------------------------
// Integers compare exactly. Floating point compares with a combined absolute /
// relative tolerance — GPU and CPU may legitimately differ in the last ULPs due
// to fused-multiply-add, different rounding of transcendentals, etc. The default
// tolerance is deliberately tight so real bugs are not masked.

struct Tolerance {
    double abs = 1e-6;   // absolute tolerance
    double rel = 1e-6;   // relative tolerance
};

template <class T>
inline bool elements_equal(T expected, T actual, const Tolerance& tol, double& err_out) {
    if constexpr (std::is_floating_point_v<T>) {
        const double e = static_cast<double>(expected);
        const double a = static_cast<double>(actual);
        if (std::isnan(e) && std::isnan(a)) { err_out = 0.0; return true; }
        const double diff = std::fabs(e - a);
        err_out = diff;
        const double allowed = tol.abs + tol.rel * std::fabs(e);
        return diff <= allowed;
    } else {
        err_out = (expected == actual) ? 0.0 : 1.0;
        return expected == actual;
    }
}

// ---------------------------------------------------------------------------
// Input generation (seeded, reproducible)
// ---------------------------------------------------------------------------

class Rng {
public:
    explicit Rng(std::uint64_t seed) : engine_(seed) {}

    template <class T>
    T value(T lo, T hi) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::uniform_real_distribution<T>(lo, hi)(engine_);
        } else {
            return std::uniform_int_distribution<std::int64_t>(
                       static_cast<std::int64_t>(lo), static_cast<std::int64_t>(hi))(engine_);
        }
    }

    std::mt19937_64& engine() { return engine_; }

private:
    std::mt19937_64 engine_;
};

template <class T>
std::vector<T> make_input(std::size_t n, Rng& rng, T lo, T hi) {
    std::vector<T> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) v.push_back(rng.value<T>(lo, hi));
    return v;
}

// The canonical size sweep: edge cases (empty, single, sub-workgroup) plus sizes
// that straddle the 256-thread workgroup boundary and a large case.
inline const std::vector<std::size_t>& default_sizes() {
    static const std::vector<std::size_t> sizes = {
        0, 1, 2, 7, 31, 32, 33, 255, 256, 257, 1000, 4096, 100000};
    return sizes;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

class Report {
public:
    explicit Report(std::string suite) : suite_(std::move(suite)) {
        std::cout << "[ RUN  ] " << suite_ << "\n";
    }

    // Compare two equal-length result containers. Returns true on match.
    template <class T>
    bool compare(const std::string& case_name, std::size_t n,
                 const std::vector<T>& expected, const std::vector<T>& actual,
                 const Tolerance& tol = {}) {
        ++total_;
        if (expected.size() != actual.size()) {
            fail(case_name, n);
            std::cout << "    size mismatch: expected " << expected.size()
                      << " got " << actual.size() << "\n";
            return false;
        }
        std::size_t mismatches = 0;
        std::size_t first_idx = 0;
        double first_err = 0.0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            double err = 0.0;
            if (!elements_equal(expected[i], actual[i], tol, err)) {
                if (mismatches == 0) { first_idx = i; first_err = err; }
                ++mismatches;
            }
        }
        if (mismatches == 0) {
            ++passed_;
            return true;
        }
        fail(case_name, n);
        std::ostringstream os;
        os << "    " << mismatches << "/" << expected.size() << " elements differ; "
           << "first at [" << first_idx << "] expected=" << std::setprecision(9)
           << +expected[first_idx] << " actual=" << +actual[first_idx]
           << " err=" << first_err << "\n";
        std::cout << os.str();
        return false;
    }

    // Compare two scalar results (for reduce-like algorithms).
    template <class T>
    bool compare_scalar(const std::string& case_name, std::size_t n,
                        T expected, T actual, const Tolerance& tol = {}) {
        ++total_;
        double err = 0.0;
        if (elements_equal(expected, actual, tol, err)) { ++passed_; return true; }
        fail(case_name, n);
        std::cout << "    scalar differs: expected=" << std::setprecision(9) << +expected
                  << " actual=" << +actual << " err=" << err << "\n";
        return false;
    }

    int finish() {
        const bool ok = (passed_ == total_);
        std::cout << "[ " << (ok ? "PASS" : "FAIL") << " ] " << suite_ << " — "
                  << passed_ << "/" << total_ << " cases passed\n";
        return ok ? 0 : 1;
    }

private:
    void fail(const std::string& case_name, std::size_t n) {
        std::cout << "  [FAIL] " << case_name << " (n=" << n << ")\n";
    }

    std::string suite_;
    std::size_t total_ = 0;
    std::size_t passed_ = 0;
};

// ---------------------------------------------------------------------------
// Differential drivers
// ---------------------------------------------------------------------------
// Each driver takes a "kernel" callable that applies the operation under a given
// execution policy, runs it under seq and par on identical copies, and diffs.

// In-place transform-style algorithms (for_each, transform-in-place, fill, ...):
// kernel signature is  void(ExecutionPolicy, std::vector<T>&)
template <class T, class Kernel>
void diff_inplace(Report& report, const std::string& case_name,
                  const std::vector<T>& input, Kernel kernel,
                  const Tolerance& tol = {}) {
    std::vector<T> a = input;  // ground truth (seq)
    std::vector<T> b = input;  // candidate (par)
    kernel(std::execution::seq, a);
    kernel(std::execution::par, b);
    report.compare(case_name, input.size(), a, b, tol);
}

// Algorithms producing an output container: kernel signature is
//   void(ExecutionPolicy, const std::vector<T>& in, std::vector<U>& out)
template <class T, class U, class Kernel>
void diff_out(Report& report, const std::string& case_name,
              const std::vector<T>& input, Kernel kernel,
              const Tolerance& tol = {}) {
    std::vector<U> a(input.size());
    std::vector<U> b(input.size());
    kernel(std::execution::seq, input, a);
    kernel(std::execution::par, input, b);
    report.compare(case_name, input.size(), a, b, tol);
}

// Reduction-style algorithms producing a scalar: kernel signature is
//   T(ExecutionPolicy, const std::vector<T>&)
template <class T, class Kernel>
void diff_scalar(Report& report, const std::string& case_name,
                 const std::vector<T>& input, Kernel kernel,
                 const Tolerance& tol = {}) {
    const T a = kernel(std::execution::seq, input);
    const T b = kernel(std::execution::par, input);
    report.compare_scalar(case_name, input.size(), a, b, tol);
}

}  // namespace pcts
