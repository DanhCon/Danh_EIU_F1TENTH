cmake_minimum_required(VERSION 3.8)
project(ros2_f1tenth_basic)

find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rcutils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ackermann_msgs REQUIRED)





install(PROGRAMS 
  src/simple_odom_tb3.py
  src/tb3_wheel_odometry.py
  src/on_lai_sub.py
  src/on_lai_pub.py
  src/service_server.py
  src/service_client.py
  src/sub_odom.py
  src/pp_basic.py
  src/stanley.py
  src/optimized_pp.py




  DESTINATION lib/${PROJECT_NAME})  




include_directories(
  include
  ${dynamixel_sdk_INCLUDE_DIRS}
)

add_executable(main_test
  src/main.cpp
  src/dynamixel_sdk_wrapper.cpp
)

add_executable(talker src/pub_cpp.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(cpp_pub_sub src/review_cpp_pub_sub.cpp)
ament_target_dependencies(cpp_pub_sub rclcpp std_msgs sensor_msgs ackermann_msgs)


# 3. Định nghĩa file chạy cho Subscriber
add_executable(listener src/sub_cpp.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

add_executable(robot_news src/robot_new.cpp)
ament_target_dependencies(robot_news rclcpp std_msgs)




ament_target_dependencies(main_test
  rcutils
    # nếu bạn có dùng ROS2 node
)
# === 1. THÊM ĐỂ CÀI ĐẶT FILE LAUNCH ===
# Lệnh này sẽ copy toàn bộ thư mục 'launch' vào 'install/share/motor_control/launch'
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# === 2. THÊM ĐỂ CÀI ĐẶT FILE CONFIG ===
# Lệnh này sẽ copy toàn bộ thư mục 'config' vào 'install/share/motor_control/config'
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

target_link_libraries(main_test ${dynamixel_sdk_LIBRARIES})

install(TARGETS main_test talker
  listener
  robot_news
  cpp_pub_sub


DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
