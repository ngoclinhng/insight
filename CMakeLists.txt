# Insight - a simple, fast, and flexible Machine Learning library.
#
# Author: mail2ngoclinh@gmail.com

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(VERSION 3.5)

# C++ version constraints.
# ==========================================================================

# Set the C++ version (must be >= C++11) when compiling Insight.
#
# Reflect a user-specified (via -D) CMAKE_CXX_STANDARD if present, otherwise
# default to C++11.
set(DEFAULT_CXX_STANDARD ${CMAKE_CXX_STANDARD})
if (NOT DEFAULT_CXX_STANDARD)
  set(DEFAULT_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD ${DEFAULT_CXX_STANDARD} CACHE STRING "C++ standard
  (minimum 11)" FORCE)
# Restrict CMAKE_CXX_STANDARD to the valid versions permitted and ensure that
# if one was forced via -D that it is in the valid set.
set(ALLOWED_CXX_STANDARDS 11 14 17)
set_property(CACHE CMAKE_CXX_STANDARD PROPERTY STRINGS ${ALLOWED_CXX_STANDARDS})
list(FIND ALLOWED_CXX_STANDARDS ${CMAKE_CXX_STANDARD} POSITION)
if (POSITION LESS 0)
  message(FATAL_ERROR "Invalid CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}. "
    "Must be one of: ${ALLOWED_CXX_STANDARDS}")
endif()
# Specify the standard as a hard requirement, otherwise CMAKE_CXX_STANDARD is
# interpreted as a suggestion that can decay *back* to lower versions.
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "")
mark_as_advanced(CMAKE_CXX_STANDARD_REQUIRED)

# MSVC versions < 2013 did not fully support >= C++11.
if (MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "Invalid CMAKE_CXX_COMPILER_VERSION: "
    "${CMAKE_CXX_COMPILER_VERSION}. Insight requires at least MSVC 2013 Update 4+")
endif()

project(Insight C CXX)

list(APPEND CMAKE_MODULE_PATH "${Insight_SOURCE_DIR}/cmake")

# Insight output directories.
# ==========================================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${Insight_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${Insight_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${Insight_BINARY_DIR}/lib)

# Allow user to specify a suffix for the library install directory, the only
# really sensible option (other than "") being "64", such that:
# ${CMAKE_INSTALL_PREFIX}/lib -> ${CMAKE_INSTALL_PREFIX}/lib64.
#
# Heuristic for determining LIB_SUFFIX. FHS recommends that 64-bit systems
# install native libraries to lib64 rather than lib. Most distros seem to
# follow this convention with a couple notable exceptions (Debian-based and
# Arch-based distros) which we try to detect here.
if (CMAKE_SYSTEM_NAME MATCHES "Linux" AND
    NOT DEFINED LIB_SUFFIX AND
    NOT CMAKE_CROSSCOMPILING AND
    CMAKE_SIZEOF_VOID_P EQUAL "8" AND
    NOT EXISTS "/etc/debian_version" AND
    NOT EXISTS "/etc/arch-release")
  message("-- Detected non-Debian/Arch-based 64-bit Linux distribution. "
    "Defaulting to library install directory: lib${LIB_SUFFIX}. You can "
    "override this by specifying LIB_SUFFIX.")
  set(LIB_SUFFIX "64")
endif ()
# Only create the cache variable (for the CMake GUI) after attempting to detect
# the suffix *if not specified by the user* (NOT DEFINED LIB_SUFFIX in if())
# s/t the user could override our autodetected suffix with "" if desired.
set(LIB_SUFFIX "${LIB_SUFFIX}" CACHE STRING
  "Suffix of library install directory (to support lib/lib64)." FORCE)

set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "-debug")

# Read the Insight version from the source.
include(ExtractInsightVersionFromSource)
extract_insight_version_from_source(${Insight_SOURCE_DIR})

enable_testing()

option(BUILD_TESTING "Enable tests." ON)
option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_SHARED_LIBS "Build Insight as a shared library." ON)

unset(INSIGHT_COMPILE_OPTIONS)

include(PrettyPrintCMakeList)
include(UpdateCacheVariable)

# Insight dependencies.
# ==========================================================================

# Glog.
find_package(Glog REQUIRED)

# BLAS backend for Insight.
include(InsightBLAS)
# List of all available BLAS options that Insight supports on thus platform.
find_available_blas_options(INSIGHT_AVAILABLE_BLAS_OPTIONS)
pretty_print_cmake_list(PRETTY_INSIGHT_AVAILABLE_BLAS_OPTIONS
  ${INSIGHT_AVAILABLE_BLAS_OPTIONS})
message(STATUS "Detected available BLAS options:
  ${PRETTY_INSIGHT_AVAILABLE_BLAS_OPTIONS}")

set(INSIGHT_BLAS_OPTION "${INSIGHT_BLAS_OPTION}" CACHE STRING
  "Insight BLAS backend" FORCE)

# There is no need to check for validity of INSIGHT_BLAS_OPTION which was
# specified via -DINSIGHT_BLAS_OPTION here becuase we've already done that
# in set_sapien_blas_library macro (see below).
if (NOT INSIGHT_BLAS_OPTION)
  # Defaults to the first item in the list of available options.
  list(GET INSIGHT_AVAILABLE_BLAS_OPTIONS 0 DEFAULT_BLAS_OPTION)
  update_cache_variable(INSIGHT_BLAS_OPTION ${DEFAULT_BLAS_OPTION})
endif()

set_insight_blas_library("${INSIGHT_BLAS_OPTION}")

if (BUILD_SHARED_LIBS)
  message(STATUS "Building Insight as a shared library.")

  # The INSIGHT_BUILDING_SHARED_LIBRARY compile option is NOT stored in the
  # INSIGHT_COMPILE_OPTIONS as it must only be defined when Insight is
  # compiled not when it is used as it controls the INSIGHT_EXPORT macro.
  add_definitions(-DINSIGHT_BUILDING_SHARED_LIBRARY)
  list(APPEND INSIGHT_COMPILE_OPTIONS INSIGHT_USING_SHARED_LIBRARY)
else()
  message(STATUS "Building Insight as a static library.")
endif()

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose built type, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
else()
  message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}")
endif()

if (UNIX)
  # GCC is not strict enough by default, so enable most of the warnings.
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
endif (UNIX)

# Alternative to malloc.
# -------------------------------------------------------------------------

include(InsightFindAlternativesToMalloc)
insight_find_alternatives_to_malloc(INSIGHT_ALTERNATIVES_TO_MALLOC_LIST)

if (INSIGHT_ALTERNATIVES_TO_MALLOC_LIST)
  pretty_print_cmake_list(PRETTY_INSIGHT_ALTERNATIVES_TO_MALLOC_LIST
    ${INSIGHT_ALTERNATIVES_TO_MALLOC_LIST})
  message(STATUS "Found alternatives to malloc: ${PRETTY_INSIGHT_ALTERNATIVES_TO_MALLOC_LIST}")

  set(INSIGHT_MALLOC_OPTION "${INSIGHT_MALLOC_OPTION}" CACHE STRING
    "Alternative to C malloc" FORCE)

  # There is no need to check for validity of INSIGHT_MALLOC_OPTION which was
  # specified via -DINSIGHT_MALLOC_OPTION here becuase we've already done
  # that in set_insight_malloc_option macro (see below).
  if (NOT INSIGHT_MALLOC_OPTION)
    # Defaults to the first item in the list of available options.
    list(GET INSIGHT_ALTERNATIVES_TO_MALLOC_LIST 0 DEFAULT_MALLOC_OPTION)
    update_cache_variable(INSIGHT_MALLOC_OPTION ${DEFAULT_MALLOC_OPTION})
  endif()
  set_insight_malloc_option("${INSIGHT_MALLOC_OPTION}")
endif()

# Configre the Insight config.h compile options header using the current
# compile options and put the configured header into the Insight build
# directory.
list(REMOVE_DUPLICATES INSIGHT_COMPILE_OPTIONS)
include(CreateInsightConfig)
create_insight_config("${INSIGHT_COMPILE_OPTIONS}"
  ${Insight_BINARY_DIR}/config/insight/internal)

add_subdirectory(internal/insight)
  
if (BUILD_EXAMPLES)
  message(STATUS "Build the examples")
  add_subdirectory(examples)
else (BUILD_EXAMPLES)
  message(STATUS "Do not build any example.")
endif (BUILD_EXAMPLES)

# Install root public headers.
file(GLOB INSIGHT_PUBLIC_HEADER_FILES
  ${Insight_SOURCE_DIR}/include/insight/*.h)
install(FILES ${INSIGHT_PUBLIC_HEADER_FILES} DESTINATION include/insight)
file(GLOB INSIGHT_PUBLIC_INTERNAL_HEADER_FILES
  ${Insight_SOURCE_DIR}/include/insight/internal/*.h)
install(FILES ${INSIGHT_PUBLIC_INTERNAL_HEADER_FILES}
  DESTINATION include/insight/internal)

# Install linalg headers.

file(GLOB INSIGHT_PUBLIC_LINALG_HEADERS
  ${Insight_SOURCE_DIR}/include/insight/linalg/*.h)
install(FILES ${INSIGHT_PUBLIC_LINALG_HEADERS}
  DESTINATION include/insight/linalg)

file(GLOB INSIGHT_PUBLIC_LINALG_DETAIL_HEADERS
  ${Insight_SOURCE_DIR}/include/insight/linalg/detail/*.h)
install(FILES ${INSIGHT_PUBLIC_LINALG_DETAIL_HEADERS}
  DESTINATION include/insight/linalg/detail)

# Also setup installation of Insight config.h configured with the current
# build options into the installed headers directory.
install(FILES ${Insight_BINARY_DIR}/config/insight/internal/config.h
        DESTINATION include/insight/internal)

