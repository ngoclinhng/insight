# BLAS backend options for Insight
# TODO(Linh): Support MKL
set(INSIGHT_BLAS_OPTIONS "OpenBLAS;Atlas;Accelerate")

function(find_available_blas_options AVAILABLE_BLAS_OPTIONS_RESULT)
  set(AVAILABLE_BLAS_OPTIONS ${INSIGHT_BLAS_OPTIONS})

  # MKL
  # find_package(MKL QUIET)
  # if (NOT MKL_FOUND)
  #   list(REMOVE_ITEM AVAILABLE_BLAS_OPTIONS "MKL")
  # endif()

  # OpenBLAS
  find_package(OpenBLAS QUIET)
  if (NOT OpenBLAS_FOUND)
    list(REMOVE_ITEM AVAILABLE_BLAS_OPTIONS "OpenBLAS")
  endif()

  # Atlas
  find_package(Atlas QUIET)
  if (NOT Atlas_FOUND)
    list(REMOVE_ITEM AVAILABLE_BLAS_OPTIONS "Atlas")
  endif()

  # ACCELERATE/vecLib?
  find_package(vecLib QUIET)
  if (NOT VECLIB_FOUND)
    list(REMOVE_ITEM AVAILABLE_BLAS_OPTIONS "Accelerate")
  endif()

  if (NOT AVAILABLE_BLAS_OPTIONS)
    # At least one BLAS library must be chosen.
    message(FATAL_ERROR "Cannot find any BLAS library that Insight supports
      on this system")
  endif()

  set(${AVAILABLE_BLAS_OPTIONS_RESULT} ${AVAILABLE_BLAS_OPTIONS}
    PARENT_SCOPE)
endfunction()

unset(INSIGHT_BLAS_INCLUDE_DIRS)
unset(INSIGHT_BLAS_LIBRARIES)

macro(set_insight_blas_library INSIGHT_BLAS_LIBRARY_TO_SET)
  if ("${INSIGHT_BLAS_LIBRARY_TO_SET}" STREQUAL "OpenBLAS")
    # TODO(Linh): Do we reall need to find it again here
    find_package(OpenBLAS REQUIRED)
    set(INSIGHT_BLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    set(INSIGHT_BLAS_LIBRARIES ${OpenBLAS_LIB})
    # We donot need to define compile option here
  elseif ("${INSIGHT_BLAS_LIBRARY_TO_SET}" STREQUAL "Atlas")
    # TODO(Linh): Do we really need to find it again?
    find_package(Atlas REQUIRED)
    set(INSIGHT_BLAS_INCLUDE_DIRS ${Atlas_INCLUDE_DIR})
    set(INSIGHT_BLAS_LIBRARIES ${Atlas_LIBRARIES})
    # We donnot need to define compile option here as well.
  
  elseif ("${INSIGHT_BLAS_LIBRARY_TO_SET}" STREQUAL "Accelerate")
    # But we do really need to find it again here since find_package(vecLib)
    # will yield either vecLib as a standalone framwork which we donot have
    # to include Accelerate/Accelerate.h in our code, or as part of the
    # Accelerate framework which we have.
    find_package(vecLib REQUIRED)
    
    set(INSIGHT_BLAS_INCLDUE_DIRS ${vecLib_INCLUDE_DIR})
    set(INSIGHT_BLAS_LIBRARIES ${vecLib_LINKER_LIBS})
    
    # We only define INSIGHT_USE_ACCELERATE (so that we know to include
    # Accelerate/Accelerate.h in our code) when vecLib is a part of
    # Accelerate framework.
    if(NOT vecLib_INCLUDE_DIR MATCHES
      "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND INSIGHT_COMPILE_OPTIONS INSIGHT_USE_ACCELERATE)
    endif()
  else()
    include(PrettyPrintCMakeList)
    find_available_blas_options(_AVAILABLE_BLAS_OPTIONS)
    pretty_print_cmake_list(_AVAILABLE_BLAS_OPTIONS ${_AVAILABLE_BLAS_OPTIONS})
    message(FATAL_ERROR "Unknown BLAS option: '${INSIGHT_BLAS_LIBRARY_TO_SET}'"
      ". Available BLAS options are: ${_AVAILABLE_BLAS_OPTIONS}")
  endif()
  message(STATUS " Using BLAS option: ${INSIGHT_BLAS_LIBRARY_TO_SET}")
endmacro()