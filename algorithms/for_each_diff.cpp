// Differential conformance tests for std::for_each(std::execution::par, ...)
//
// These exercise the lambda shapes the Parallax compiler claims to support:
// simple arithmetic, captured scalars, and compound assignment. Built without
// the plugin, par == seq on the CPU and every case must pass (harness self-test).
// Built with the plugin, any failure is a GPU-offload correctness bug.

#include "parallax_cts/diff_harness.hpp"

#include <algorithm>
#include <execution>

int main() {
    pcts::Report report("for_each.float");
    pcts::Rng rng(0xC0FFEE);

    for (std::size_t n : pcts::default_sizes()) {
        auto input = pcts::make_input<float>(n, rng, -100.0f, 100.0f);

        // x = x * 2 + 1   (the canonical README example)
        pcts::diff_inplace<float>(report, "affine", input, [](const auto& pol, auto& v) {
            std::for_each(pol, v.begin(), v.end(), [](float& x) { x = x * 2.0f + 1.0f; });
        });

        // compound assignment operators
        pcts::diff_inplace<float>(report, "mul_assign", input, [](const auto& pol, auto& v) {
            std::for_each(pol, v.begin(), v.end(), [](float& x) { x *= 3.0f; });
        });

        // captured scalar by value
        const float k = 1.5f;
        pcts::diff_inplace<float>(report, "capture_scalar", input, [k](const auto& pol, auto& v) {
            std::for_each(pol, v.begin(), v.end(), [k](float& x) { x += k; });
        });

        // control flow inside the kernel (if) — supported per the compiler audit
        pcts::diff_inplace<float>(report, "branch", input, [](const auto& pol, auto& v) {
            std::for_each(pol, v.begin(), v.end(), [](float& x) {
                if (x > 0.0f) x *= 2.0f; else x -= 1.0f;
            });
        });

        // a bounded loop with a local accumulator — exercises DeclStmt + ForStmt
        pcts::diff_inplace<float>(report, "loop_accum", input, [](const auto& pol, auto& v) {
            std::for_each(pol, v.begin(), v.end(), [](float& x) {
                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) sum += static_cast<float>(i);
                x += sum;  // x + 6
            });
        });
    }

    return report.finish();
}
