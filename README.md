# Parallax Conformance Test Suite (CTS)

Differential conformance tests for the Parallax GPU-offload backend.

## How it works

Every test runs the **same** standard C++ algorithm twice on identical inputs:

| Role | Policy | Where it runs |
|------|--------|---------------|
| Ground truth | `std::execution::seq` | CPU, serial |
| Candidate | `std::execution::par` | GPU (with the Parallax plugin) / CPU (without) |

The two results are diffed elementwise (exact for integers, tolerance-based for
floating point) across a sweep of sizes — including the empty, single-element,
and workgroup-boundary (255/256/257) edge cases — over fuzzed values. Any
divergence is a real GPU-offload correctness bug.

The harness is pure ISO C++ (header-only, `include/parallax_cts/diff_harness.hpp`)
with no dependency on the runtime, so the exact same test source compiles in both
modes below.

## Build modes

### Self-test (default, no GPU required)

Both policies run on the CPU, so every non-gap test must pass. This validates the
harness and the test kernels and runs anywhere.

```bash
cmake -S . -B build-selftest
cmake --build build-selftest -j
cd build-selftest && ctest --output-on-failure
```

### Plugin (real GPU conformance)

```bash
cmake -S . -B build-plugin \
  -DPARALLAX_PLUGIN=/path/to/libparallax-clang-plugin.so \
  -DPARALLAX_RUNTIME_DIR=/path/to/parallax-runtime/build
cmake --build build-plugin -j
cd build-plugin && ctest --output-on-failure
```

`std::execution::par` is now offloaded to the GPU and the diffs become real
conformance checks.

## Test files

| File | Covers |
|------|--------|
| `algorithms/for_each_diff.cpp` | `for_each`: arithmetic, captures, branches, loops, local vars |
| `algorithms/transform_diff.cpp` | `transform`: arithmetic, `sqrt`, branches |
| `algorithms/known_gaps_diff.cpp` | **Known bugs**, encoded as executable tests (see below) |

## Known-gap tests

`known_gaps_diff.cpp` encodes current, documented limitations as runnable tests.
They pass in self-test mode (the CPU is correct) and are marked `WILL_FAIL` in
plugin mode. When a gap is fixed, its test flips to "unexpectedly passed" and we
remove the `GAP` marker in `algorithms/CMakeLists.txt`. Current entries:

- **int64 truncation** — 64-bit integers are silently narrowed to 32-bit in the
  SPIR-V generator; values above 2³¹ lose their high bits.
- **`<cmath>` intrinsics** — only `sqrt` is mapped; other math functions emit a
  placeholder constant.

See `PARALLAX_STDPAR_COMPLIANCE_PLAN.md` (repo root) for the full roadmap.

## Adding a test

```cpp
#include "parallax_cts/diff_harness.hpp"
int main() {
    pcts::Report report("my_suite");
    pcts::Rng rng(0xSEED);
    for (std::size_t n : pcts::default_sizes()) {
        auto input = pcts::make_input<float>(n, rng, lo, hi);
        pcts::diff_inplace<float>(report, "case_name", input,
            [](const auto& pol, auto& v) { /* std::algo(pol, ...) */ });
    }
    return report.finish();
}
```

Then register it in `algorithms/CMakeLists.txt` with `pcts_add_test`.
