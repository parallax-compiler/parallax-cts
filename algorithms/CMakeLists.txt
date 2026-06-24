# Differential conformance tests, one executable per algorithm.
pcts_add_test(for_each_diff   for_each_diff.cpp)
pcts_add_test(transform_diff  transform_diff.cpp)

# Tests that encode known current limitations. Expected to fail under the plugin
# until the corresponding gap is closed (see PARALLAX_STDPAR_COMPLIANCE_PLAN.md).
pcts_add_test(known_gaps_diff known_gaps_diff.cpp GAP)
