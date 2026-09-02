cmake_minimum_required(VERSION 3.8)
project(aruco_ros_cv)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(aruco_msgs REQUIRED)
find_package(aruco_ros_cv_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Common dependencies list (no shared library — header-only approach)
set(COMMON_DEPS
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  aruco_msgs
  cv_bridge
  OpenCV
)

# ---- Helper macro: add a node executable ----
macro(add_aruco_node TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  target_include_directories(${TARGET_NAME} PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
  )
  ament_target_dependencies(${TARGET_NAME} ${ARGN})
  target_link_libraries(${TARGET_NAME} ${OpenCV_LIBRARIES})
endmacro()

# ---- Part 1: Service node ----
add_aruco_node(aruco_detect_service
  src/aruco_detect_service.cpp
  ${COMMON_DEPS} aruco_ros_cv_interfaces
)

# ---- Part 2: Single image detection node ----
add_aruco_node(single_aruco_image_detection
  src/single_aruco_image_detection.cpp
  ${COMMON_DEPS}
)

# ---- Part 3: RT capture node ----
add_aruco_node(aruco_rt_capture
  src/aruco_rt_capture.cpp
  ${COMMON_DEPS} image_transport tf2 tf2_ros tf2_geometry_msgs
)

# ---- Part 4: Action server node ----
add_aruco_node(aruco_action_server
  src/aruco_action_server.cpp
  ${COMMON_DEPS} rclcpp_action aruco_ros_cv_interfaces diagnostic_msgs tf2 tf2_ros tf2_geometry_msgs
)

# ---- Part 5: Camera calibration node ----
add_aruco_node(camera_calibration
  src/camera_calibration.cpp
  ${COMMON_DEPS}
)

# ---- Part 6: Start-detection CLI client node ----
add_aruco_node(aruco_start_detection
  src/aruco_start_detection.cpp
  ${COMMON_DEPS} rclcpp_action aruco_ros_cv_interfaces
)

# ---- Part 7: Stop-detection CLI client node ----
add_aruco_node(aruco_stop_detection
  src/aruco_stop_detection.cpp
  ${COMMON_DEPS} aruco_ros_cv_interfaces
)

# ---- Install ----
install(TARGETS
  aruco_detect_service
  single_aruco_image_detection
  aruco_rt_capture
  aruco_action_server
  camera_calibration
  aruco_start_detection
  aruco_stop_detection
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY resources/
  DESTINATION share/${PROJECT_NAME}/resources
)

ament_package()
