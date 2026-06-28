cmake_minimum_required(VERSION 3.8)
project(motor_control)

find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rcutils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(turtlesim REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})


install(DIRECTORY include 

DESTINATION include)



install(PROGRAMS 
  src/control_f1tenth_basic/simple_odom_tb3.py
  src/control_f1tenth_basic/tb3_wheel_odometry.py
  src/ros_learning_basic/on_lai_sub.py
  src/ros_learning_basic/on_lai_pub.py
  src/ros_learning_basic/service_server.py
  src/ros_learning_basic/service_client.py
  src/control_f1tenth_basic/sub_odom.py
  src/control_f1tenth_basic/pp_basic.py
  src/control_f1tenth_basic/stanley.py
  src/control_f1tenth_basic/optimized_pp.py
  src/PP_rrt/rrt.py
  src/PP_rrt/rrt_plus_pp.py
  src/TF/simple_tf_kinematics.py
  src/learning_odom_project1_py/control_speed_basic.py
  src/learning_odom_project1_py/rua_chay_bo.py




  DESTINATION lib/${PROJECT_NAME})  




include_directories(
  include
  ${dynamixel_sdk_INCLUDE_DIRS}
)

add_executable(main_test
  src/cpp_learning/main.cpp
  src/cpp_learning/dynamixel_sdk_wrapper.cpp
)

add_executable(simple_turtlesim_kinematics
  src/learning_odom_project1_cpp/simple_turtlesim_kinematics.cpp

)


add_executable(simple_controller
  src/learning_odom_project1_cpp/simple_controller.cpp

)
ament_target_dependencies(simple_controller rclcpp std_msgs geometry_msgs  ${Eigen_LIBRARIES})


ament_target_dependencies(simple_turtlesim_kinematics rclcpp turtlesim)



add_executable(talker src/cpp_learning/pub_cpp.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

# 3. Định nghĩa file chạy cho Subscriber
add_executable(listener src/cpp_learning/sub_cpp.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

add_executable(robot_news src/cpp_learning/robot_new.cpp)
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
  simple_turtlesim_kinematics
  simple_controller


DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()






