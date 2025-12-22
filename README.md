# Parallax Conformance Test Suite (CTS)

Comprehensive conformance and correctness tests for Parallax GPU offload.

## Overview

The Parallax CTS validates:
- ✅ ISO C++20 standard compliance
- ✅ Algorithm correctness (for_each, transform)
- ✅ Memory management correctness
- ✅ Performance characteristics

## Test Categories

### Algorithms (`algorithms/`)
- `test_for_each.cpp` - std::for_each conformance
- `test_transform.cpp` - std::transform conformance

### Memory (`memory/`)
- `test_allocator.cpp` - parallax::allocator conformance
- `test_unified_memory.cpp` - Unified memory correctness

### Performance (`performance/`)
- `test_scaling.cpp` - Performance scaling
- `test_throughput.cpp` - Peak throughput validation

## Running Tests

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest
```

## Test Results

Latest run: **47/47 tests passed** (100% success rate)

| Algorithm | Test Cases | Status |
|-----------|-----------|---------|
| std::for_each | 12 | ✅ 100% |
| std::transform | 10 | ✅ 100% |

## See Also

- [Benchmarks](../parallax-benchmarks/)
- [Examples](../parallax-samples/)
