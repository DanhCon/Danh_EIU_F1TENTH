cmake_minimum_required(VERSION 3.8)
project(ros2_f1tenth_basic)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosbasic_msgs REQUIRED)


install(PROGRAMS
  learning_ros/publisher.py    # tao folder: lib , goi ros_publisher.py
    # tao folder: lib , goi ros_subscriber.py
  
  learning_ros/move_robot.py
  learning_ros/service.py
  learning_ros/client.py
  learning_ros/transform.py
  learning_ros/break.py
  learning_ros/safety.py
  Visual/visual.py
  learning_ros/chay2m.py
  learning_ros/go_lai.py
  learning_ros/go_lai_2.py


  DESTINATION lib/${PROJECT_NAME})  


install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})


install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
