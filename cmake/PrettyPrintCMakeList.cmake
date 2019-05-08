# pretty_print_cmake_list(OUTPUT_VAR [item1 [item2 ...]])
#
# Sets ${OUTPUT_VAR} in the caller's scope to a human-readable string
# representation of the list passed as the ramaining arguments formed
# as: "[item1, item2, ..., itemN]".
function(pretty_print_cmake_list OUTPUT_VAR)
  string(REPLACE ";" ", " PRETTY_LIST_STRING "[${ARGN}]")
  set(${OUTPUT_VAR} "${PRETTY_LIST_STRING}" PARENT_SCOPE)
endfunction()