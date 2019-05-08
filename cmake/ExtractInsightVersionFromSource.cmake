# Extract Insight version from the version file:
#
#   <Insight_SOURCE_DIR>/include/insight/version.h

macro(extract_insight_version_from_source INSIGHT_SOURCE_ROOT)
  set(INSIGHT_VERSION_FILE
    ${INSIGHT_SOURCE_ROOT}/include/insight/version.h)
  if (NOT EXISTS ${INSIGHT_VERSION_FILE})
    message(FATAL_ERROR "Cannot find Insight version.h file in specified "
      " Insight source directory: ${INSIGHT_SOURCE_ROOT}")
  endif()

  file(READ ${INSIGHT_VERSION_FILE} INSIGHT_VERSION_FILE_CONTENTS)

  # Extract major version.
  string(REGEX MATCH "#define INSIGHT_VERSION_MAJOR [0-9]+"
    INSIGHT_VERSION_MAJOR "${INSIGHT_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define INSIGHT_VERSION_MAJOR ([0-9]+)" "\\1"
    INSIGHT_VERSION_MAJOR "${INSIGHT_VERSION_MAJOR}")
  if ("${INSIGHT_VERSION_MAJOR}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract Insight major version from "
      "${INSIGHT_VERSION_FILE}")
  endif()

  # Extract minor version.
  string(REGEX MATCH "#define INSIGHT_VERSION_MINOR [0-9]+"
    INSIGHT_VERSION_MINOR "${INSIGHT_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define INSIGHT_VERSION_MINOR ([0-9]+)" "\\1"
    INSIGHT_VERSION_MINOR "${INSIGHT_VERSION_MINOR}")
  if ("${INSIGHT_VERSION_MINOR}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract Insight minor version from "
      "${INSIGHT_VERSION_FILE}")
  endif()

  # Extract patch version.
  string(REGEX MATCH "#define INSIGHT_VERSION_REVISION [0-9]+"
    INSIGHT_VERSION_PATCH "${INSIGHT_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define INSIGHT_VERSION_REVISION ([0-9]+)" "\\1"
    INSIGHT_VERSION_PATCH "${INSIGHT_VERSION_PATCH}")
  if ("${INSIGHT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract Insight patch version from "
      "${INSIGHT_VERSION_FILE}")
  endif()

  # The full version x.x.x
  set(INSIGHT_VERSION "${INSIGHT_VERSION_MAJOR}.${INSIGHT_VERSION_MINOR}.${INSIGHT_VERSION_PATCH}")

  # Report
  message(STATUS "Detected Insight version: ${INSIGHT_VERSION} from "
    "${INSIGHT_VERSION_FILE}")
endmacro()    