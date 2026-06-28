# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_waypoint_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED waypoint_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(waypoint_FOUND FALSE)
  elseif(NOT waypoint_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(waypoint_FOUND FALSE)
  endif()
  return()
endif()
set(_waypoint_CONFIG_INCLUDED TRUE)

# output package information
if(NOT waypoint_FIND_QUIETLY)
  message(STATUS "Found waypoint: 0.0.0 (${waypoint_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'waypoint' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${waypoint_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(waypoint_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${waypoint_DIR}/${_extra}")
endforeach()
