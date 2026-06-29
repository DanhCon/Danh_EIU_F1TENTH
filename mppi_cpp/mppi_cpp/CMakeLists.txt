cmake_minimum_required(VERSION 3.8)
project(mppi_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# OpenMP for parallelizing MPPI rollouts
find_package(OpenMP REQUIRED)

add_executable(mppi_real_cpp src/mppi_real.cpp)
target_compile_options(mppi_real_cpp PRIVATE -O3 -ffast-math)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mppi_real_cpp OpenMP::OpenMP_CXX)
endif()
target_include_directories(mppi_real_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(mppi_real_cpp
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  ackermann_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(mppi_sim_cpp src/mppi_sim.cpp)
target_compile_options(mppi_sim_cpp PRIVATE -O3 -ffast-math)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mppi_sim_cpp OpenMP::OpenMP_CXX)
endif()
target_include_directories(mppi_sim_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(mppi_sim_cpp
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  ackermann_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

install(TARGETS
  mppi_real_cpp
  mppi_sim_cpp
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
