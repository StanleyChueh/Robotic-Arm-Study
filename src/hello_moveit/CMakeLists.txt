cmake_minimum_required(VERSION 3.8)
project(hello_moveit)

# Enable compiler warnings (optional)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(graph_msgs REQUIRED)   # <-- Added this line
find_package(moveit_visual_tools REQUIRED)

# Add the executable
add_executable(hello_moveit src/hello_moveit.cpp)

# Include directories
target_include_directories(hello_moveit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${moveit_ros_planning_interface_INCLUDE_DIRS}
  ${graph_msgs_INCLUDE_DIRS}         # <-- Ensure graph_msgs include directories are added
)

# Set the C++ standard
target_compile_features(hello_moveit PUBLIC cxx_std_17)

# Link dependencies
ament_target_dependencies(
  hello_moveit
  rclcpp
  moveit_ros_planning_interface
  graph_msgs           # <-- Link graph_msgs dependency
  moveit_visual_tools
)

# Install target
install(TARGETS hello_moveit
  DESTINATION lib/${PROJECT_NAME})

# Test configuration (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
