cmake_minimum_required(VERSION 3.8)
project(drl_nav_worlds)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch maps models worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_flake8_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../ament_flake8.ini")
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
