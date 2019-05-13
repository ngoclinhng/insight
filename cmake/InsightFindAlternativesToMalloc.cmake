function(insight_find_alternatives_to_malloc INSIGHT_ALTERNATIVES_TO_MALLOC)
  set(INSIGHT_POSIBLE_MALLOC_OPTIONS
    "scalable_malloc;mkl_malloc;posix_memalign")

  # TBB scalable_malloc.
  find_package(TBB QUIET)
  if (NOT TBB_MALLOC_FOUND)
    list(REMOVE_ITEM INSIGHT_POSIBLE_MALLOC_OPTIONS "scalable_malloc")
  endif()

  # Find MKL.
  find_package(MKL QUIET)
  if (NOT MKL_FOUND)
    list(REMOVE_ITEM INSIGHT_POSIBLE_MALLOC_OPTIONS "mkl_malloc")
  endif()

  # posix_memalign.
  include(CheckFunctionExists)
  check_function_exists(posix_memalign INSIGHT_HAVE_POSIX_MEMALIGN)
  if (NOT INSIGHT_HAVE_POSIX_MEMALIGN)
    list(REMOVE_ITEM INSIGHT_POSIBLE_MALLOC_OPTIONS "posix_memalign")
  endif()

  set(${INSIGHT_ALTERNATIVES_TO_MALLOC} ${INSIGHT_POSIBLE_MALLOC_OPTIONS}
    PARENT_SCOPE)
endfunction()

unset(INSIGHT_MALLOC_INCLUDE_DIRS)
unset(INSIGHT_MALLOC_LIBRARIES)

macro(set_insight_malloc_option MALLOC_OPTION_TO_SET)
  if ("${MALLOC_OPTION_TO_SET}" STREQUAL "scalable_malloc")
    find_package(TBB REQUIRED)
    set(INSIGHT_MALLOC_INCLUDE_DIRS ${TBB_MALLOC_INCLUDE_DIRS})
    set(INSIGHT_MALLOC_LIBRARIES ${TBB_MALLOC_LIBRARIES})
    list(APPEND INSIGHT_COMPILE_OPTIONS INSIGHT_USE_TBB_SCALABLE_MALLOC)
  elseif ("${MALLOC_OPTION_TO_SET}" STREQUAL "mkl_malloc")
    # TODO(Linh): This would be a redundancy if INSIGHT_BLAS_OPTION was set
    # to MKL.
    find_package(MKL REQUIRED)
    if (NOT "${INSIGHT_BLAS_OPTION}" STREQUAL "MKL")
      set(INSIGHT_MALLOC_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
      set(INSIGHT_MALLOC_LIBRARIES ${MKL_LIBRARIES})
    else()
      # If the INSIGHT_BLAS_OPTION is set to MKL and INSIGHT_MALLOC_OPTION
      # is set to mkl_malloc, then there is no need to define malloc
      # libraries here because that's already done it set blas option.
      set(INSIGHT_MALLOC_INCLUDE_DIRS "")
      set(INSIGHT_MALLOC_LIBRARIES "")
    endif()
    list(APPEND INSIGHT_COMPILE_OPTIONS INSIGHT_USE_MKL_MALLOC)
  elseif ("${MALLOC_OPTION_TO_SET}" STREQUAL "posix_memalign")
    # There is nothing to set here?
    set(INSIGHT_MALLOC_INCLUDE_DIRS "")
    set(INSIGHT_MALLOC_LIBRARIES "")
    list(APPEND INSIGHT_COMPILE_OPTIONS INSIGHT_USE_POSIX_MEMALIGN)
  else()
    include(PrettyPrintCMakeList)
    insight_find_alternatives_to_malloc(_ALTERNATIVES_TO_MALLOC)
    pretty_print_cmake_list(_ALTERNATIVES_TO_MALLOC
      ${_ALTERNATIVES_TO_MALLOC})
    message(FATAL_ERROR "Unknown malloc option: ${MALLOC_OPTION_TO_SET}. "
      "Available malloc options are: ${_ALTERNATIVES_TO_MALLOC}")
  endif()
  message(STATUS "Using malloc option: ${MALLOC_OPTION_TO_SET}")
endmacro()

