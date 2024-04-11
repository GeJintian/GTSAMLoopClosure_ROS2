cmake_minimum_required(VERSION 3.5)
project(GTSAM_loop_closure)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()
add_definitions(-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(
  include
  thirdparty/nanoflann/include
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS} 
  ${pcl_conversions_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/simple_loop_closure_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  rclpy
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
)

target_link_libraries(${PROJECT_NAME}_node
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  gtsam
)

install(TARGETS
  ${PROJECT_NAME}_node
  ARCHIVE
  LIBRARY
  RUNTIME
  DESTINATION lib/${PROJECT_NAME}
)

# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION lib
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION bin
# )

# install(TARGETS ${PROJECT_NAME}_node
#   DESTINATION lib/${PROJECT_NAME}
# )

ament_package()