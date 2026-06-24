cmake_minimum_required(VERSION 3.20)
project(parallax-cts CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ---------------------------------------------------------------------------
# Build modes
# ---------------------------------------------------------------------------
# Self-test mode (default): no plugin. Both std::execution::seq and ::par run on
#   the CPU, so every non-gap test must pass. This validates the harness itself
#   and the test kernels, and runs anywhere — no GPU required.
#
# Plugin mode: pass -DPARALLAX_PLUGIN=/path/to/libparallax-clang-plugin.{so,dylib}
#   (and -DPARALLAX_RUNTIME_DIR=/path/to/runtime/build). std::execution::par is
#   then offloaded to the GPU and the differential tests become real conformance
#   tests. Cases under known_gaps_diff are marked WILL_FAIL in this mode.
# ---------------------------------------------------------------------------
set(PARALLAX_PLUGIN "" CACHE FILEPATH "Path to the Parallax clang plugin (enables GPU offload)")
set(PARALLAX_RUNTIME_DIR "" CACHE PATH "Path to the built Parallax runtime (lib + include)")

if(PARALLAX_PLUGIN)
    set(PARALLAX_MODE "plugin")
else()
    set(PARALLAX_MODE "selftest")
endif()
message(STATUS "Parallax CTS build mode: ${PARALLAX_MODE}")

add_library(parallax_cts_harness INTERFACE)
target_include_directories(parallax_cts_harness INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# libc++ on Apple gates the parallel execution policies behind the experimental
# library. Detect and enable it so the self-test build works out of the box.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles(
    "#include <execution>\nint main(){ (void)std::execution::par; }"
    PARALLAX_HAS_EXECUTION_POLICIES)
if(NOT PARALLAX_HAS_EXECUTION_POLICIES)
    set(CMAKE_REQUIRED_FLAGS "-std=c++20 -fexperimental-library")
    check_cxx_source_compiles(
        "#include <execution>\nint main(){ (void)std::execution::par; }"
        PARALLAX_NEEDS_EXPERIMENTAL_LIB)
    if(PARALLAX_NEEDS_EXPERIMENTAL_LIB)
        target_compile_options(parallax_cts_harness INTERFACE -fexperimental-library)
        target_link_options(parallax_cts_harness INTERFACE -fexperimental-library)
    endif()
endif()

# On Linux/libstdc++ the parallel execution policies are implemented on top of
# Intel TBB, which must be present at compile and link time. Link it when found.
find_package(TBB QUIET COMPONENTS tbb)
if(TBB_FOUND)
    target_link_libraries(parallax_cts_harness INTERFACE TBB::tbb)
    message(STATUS "Parallax CTS: linking TBB for parallel execution policies")
endif()

# Helper: register one differential test executable.
#   pcts_add_test(<name> <source> [GAP])
# GAP marks the test WILL_FAIL when built in plugin mode (documents a known bug).
function(pcts_add_test name source)
    cmake_parse_arguments(ARG "GAP" "" "" ${ARGN})
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE parallax_cts_harness)

    if(PARALLAX_PLUGIN)
        target_compile_options(${name} PRIVATE -fplugin=${PARALLAX_PLUGIN})
        target_compile_definitions(${name} PRIVATE PARALLAX_ENABLED=1)
        if(PARALLAX_RUNTIME_DIR)
            target_include_directories(${name} PRIVATE ${PARALLAX_RUNTIME_DIR}/../include)
            target_link_directories(${name} PRIVATE ${PARALLAX_RUNTIME_DIR})
            target_link_libraries(${name} PRIVATE parallax-runtime)
        endif()
    endif()

    add_test(NAME ${name} COMMAND ${name})
    if(ARG_GAP AND PARALLAX_PLUGIN)
        set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
    endif()
endfunction()

add_subdirectory(algorithms)
