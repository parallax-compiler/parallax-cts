// Differential tests that encode KNOWN current limitations of the Parallax
// compiler/runtime (from the December audit). On the CPU self-test build these
// all pass. Under the plugin build they are EXPECTED TO FAIL until the
// corresponding gap is closed — CMake marks them WILL_FAIL so CI stays honest:
// the day a gap is fixed, its test flips to "unexpectedly passed" and we remove
// the WILL_FAIL marker.
//
// Each case here is a precise, executable record of a real bug, not prose.

#include "parallax_cts/diff_harness.hpp"

#include <algorithm>
#include <cstdint>
#include <execution>

int main() {
    pcts::Report report("known_gaps");
    pcts::Rng rng(0x1234);

    for (std::size_t n : pcts::default_sizes()) {
        // GAP: int64 is silently truncated to int32 in the SPIR-V generator
        // (lambda_ir_generator.cpp:78, spirv_generator.cpp:579). Values above
        // 2^31 lose their high bits on the GPU. Use values that span past 2^32.
        {
            auto input = pcts::make_input<std::int64_t>(
                n, rng, std::int64_t(1) << 40, (std::int64_t(1) << 40) + 1000);
            pcts::diff_inplace<std::int64_t>(report, "int64_no_truncation", input,
                [](const auto& pol, auto& v) {
                    std::for_each(pol, v.begin(), v.end(),
                                  [](std::int64_t& x) { x += 1; });
                });
        }

        // GAP: only sqrt is mapped as an intrinsic; other <cmath> calls produce a
        // placeholder constant (spirv_generator.cpp ~620). sin() should match CPU.
        {
            auto input = pcts::make_input<float>(n, rng, -3.0f, 3.0f);
            pcts::diff_inplace<float>(report, "cmath_intrinsic_sin", input,
                [](const auto& pol, auto& v) {
                    std::for_each(pol, v.begin(), v.end(),
                                  [](float& x) { x = std::sin(x); });
                });
        }
    }

    return report.finish();
}
