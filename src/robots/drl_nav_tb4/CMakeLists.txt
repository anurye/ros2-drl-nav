cmake_minimum_required(VERSION 3.8)
project(drl_nav_tb4)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../ament_flake8.ini")
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
