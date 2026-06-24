// Differential conformance tests for std::transform(std::execution::par, ...)
//
// Covers unary transform (the second algorithm the Parallax compiler offloads).
// Built without the plugin, par == seq on the CPU and every case must pass.
// Built with the plugin, any failure is a GPU-offload correctness bug.

#include "parallax_cts/diff_harness.hpp"

#include <algorithm>
#include <execution>

int main() {
    pcts::Report report("transform.float");
    pcts::Rng rng(0xBEEF);

    for (std::size_t n : pcts::default_sizes()) {
        auto input = pcts::make_input<float>(n, rng, 0.0f, 100.0f);

        // out = sqrt(x) * 2   (the canonical README transform example)
        pcts::diff_out<float, float>(report, "sqrt_scale", input,
            [](const auto& pol, const auto& in, auto& out) {
                std::transform(pol, in.begin(), in.end(), out.begin(),
                               [](float x) { return std::sqrt(x) * 2.0f; });
            });

        // out = x*x - x   (pure arithmetic, no intrinsics)
        pcts::diff_out<float, float>(report, "poly", input,
            [](const auto& pol, const auto& in, auto& out) {
                std::transform(pol, in.begin(), in.end(), out.begin(),
                               [](float x) { return x * x - x; });
            });

        // branch in the transform body
        pcts::diff_out<float, float>(report, "clamp", input,
            [](const auto& pol, const auto& in, auto& out) {
                std::transform(pol, in.begin(), in.end(), out.begin(),
                               [](float x) { return x > 50.0f ? 50.0f : x; });
            });
    }

    return report.finish();
}
