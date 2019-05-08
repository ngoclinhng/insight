# # Choose parallel_for based on specified INSIGHT_THREADING_MODEL
# if (INSIGHT_THREADING_MODEL STREQUAL "CXX11_THREADS")
#   set(INSIGHT_PARALLEL_FOR_SRC
#     parallel/parallel_for_cxx.cc
#     parallel/thread_pool.cc)
# elseif (INSIGHT_THREADING_MODEL STREQUAL "OPENMP")
#   set(INSIGHT_PARALLEL_FOR_SRC parallel/parallel_for_openmp.cc)
#   if (CMAKE_COMPILER_IS_GNUCXX)
#     # OpenMP in GCC requires the GNU OpenMP library.
#     list(APPEND INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES gomp)
#   endif()
# elseif (INSIGHT_THREADING_MODEL STREQUAL "NO_THREADS")
#   set(INSIGHT_PARALLEL_FOR_SRC parallel/parallel_for_nothreads.cc)
# endif()

# List all internal source files. Do NOT use file(GLOB *) to find source!
set(INSIGHT_SOURCE_FILES
  math_functions.cc
)

# Also depends on the internal header files so that they appear in IDES.
set(INSIGHT_INTERNAL_HEADER_FILES  "")
file(GLOB INSIGHT_INTERNAL_HEADER_FILES
  *.h
)

# Depend also on public headers so they appear in IDEs.
file(GLOB INSIGHT_PUBLIC_HEADER_FILES
  ${Insight_SOURCE_DIR}/include/insight/*.h)
file(GLOB INSIGHT_PUBLIC_INTERNAL_HEADER_FILES
  ${Insight_SOURCE_DIR}/include/insight/internal/*.h)

set(INSIGHT_LIBRARY_SOURCE
  ${INSIGHT_SOURCE_FILES}
  ${INSIGHT_INTERNAL_HEADER_FILES}
  ${INSIGHT_PUBLIC_HEADER_FILES}
  ${INSIGHT_PUBLIC_INTERNAL_HEADER_FILES})

add_library(insight ${INSIGHT_LIBRARY_SOURCE})
set_target_properties(insight PROPERTIES
  VERSION ${INSIGHT_VERSION}
  SOVERSION ${INSIGHT_VERSION_MAJOR})

# The ability to specify a minimum language version via cxx_std_[11,14,17]
# requires CMake >= 3.8.  Prior to that we have to specify the compiler features
# we require.
if (CMAKE_VERSION VERSION_LESS 3.8)
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_deleted_functions)
else()
  # Forward whatever C++ version Ceres was compiled with as our requirement
  # for downstream clients.
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_std_${CMAKE_CXX_STANDARD})
endif()
target_compile_features(insight PUBLIC ${REQUIRED_PUBLIC_CXX_FEATURES})


# Always build position-independent code (PIC), even when Building Insight
# as a static library so that shared library can link against it.
if (NOT WIN32 AND NOT BUILD_SHARED_LIBS)
  set_target_properties(insight PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES "")
set(INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES "")

list(APPEND INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES ${GLOG_LIBRARIES})
list(APPEND INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES ${INSIGHT_BLAS_LIBRARIES})
# list(APPEND INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES ${INSIGHT_OPENCL_LIBRARIES})

if (BUILD_SHARED_LIBS)
  # When building a shared library, mark all external libraries as PRIVATE
  # so they don't show up as a dependency.
  target_link_libraries(insight
    LINK_PUBLIC ${INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES}
    LINK_PRIVATE ${INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES})
else()
  # When building a static library, all external libraries are
  # PUBLIC(default).
  set(INSIGHT_LIBRARY_DEPENDENCIES
    ${INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES}
    ${INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES})
  target_link_libraries(insight ${INSIGHT_LIBRARY_DEPENDENCIES})
endif()

# Add the Insight headers to its target.
#
# Force the location containing the configured config.h to the front
# of the include_directories list (by default it is appended to the back)
# to ensure that if the user has an installed version of Insight in the same
# location as one of the dependencies (e.g. usr/local) that we find the
# config.h we just configured, not the (older) installed config.h
target_include_directories(insight BEFORE PUBLIC
  $<BUILD_INTERFACE:${Insight_BINARY_DIR}/config>)
target_include_directories(insight PRIVATE ${Insight_SOURCE_DIR}/internal)
target_include_directories(insight PUBLIC
  $<BUILD_INTERFACE:${Insight_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

set(INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS "")
set(INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS "")

# NOTE: The variable FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION is defined
#       in FindGlog.cmake module.
if (NOT FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION)
  # Only append glog include directories if the glog found was not a CMake
  # exported target that alredy includes them.
  list(APPEND INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS
    ${GLOG_INCLUDE_DIRS})
endif()

list(APPEND INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS
  ${INSIGHT_BLAS_INCLUDE_DIRS})

list(REMOVE_DUPLICATES INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(insight PRIVATE ${INC_DIR})
endforeach()
list(REMOVE_DUPLICATES INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${INSIGHT_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(insight PUBLIC ${INC_DIR})
endforeach()

install(TARGETS insight
        EXPORT  InsightExport
	    RUNTIME DESTINATION bin
	    LIBRARY DESTINATION lib${LIB_SUFFIX}
	    ARCHIVE DESTINATION lib${LIB_SUFFIX})


include(AppendTargetProperty)

if (BUILD_TESTING)
  add_library(gtest
    ${Insight_SOURCE_DIR}/internal/gtest/gmock_gtest_all.cc
    ${Insight_SOURCE_DIR}/internal/gtest/gmock_main.cc)

  target_include_directories(gtest PUBLIC
    ${Insight_SOURCE_DIR}/internal/gtest)

  if (BUILD_SHARED_LIBS)
    # Define gtest-specific shared library flags for compilation.
    append_target_property(gtest COMPILE_DEFINITIONS
      GTEST_CREATE_SHARED_LIBRARY)
  endif()

  target_link_libraries(gtest ${GLOG_LIBRARIES})

  macro (INSIGHT_TEST NAME)
    add_executable(${NAME}_test ${NAME}_test.cc)

    # Pull in local headers from the generated test directories when
    # insight_test() is invoked there, as well as the private headers in
    # this directory which may be referenced without the 'insight' path
    # prefix and all private dependencies that may be directly referenced.
    target_include_directories(${NAME}_test
      PUBLIC ${CMAKE_CURRENT_LIST_DIR}
             ${Insight_SOURCE_DIR}/internal
	         ${INSIGHT_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})

    target_link_libraries(${NAME}_test insight gtest)

    if (BUILD_SHARED_LIBS)
      # Define gtest-specific shared library flags for linking.
      append_target_property(${NAME}_test
        COMPILE_DEFINITIONS GTEST_LINKED_AS_SHARED_LIBRARY)
    endif()

    add_test(NAME ${NAME}_test
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test
	     --test_srcdir
	     ${Insight_SOURCE_DIR}/data)
  endmacro (INSIGHT_TEST)

  insight_test(math_functions)
endif (BUILD_TESTING)
