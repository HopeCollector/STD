include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

find_package(expr-utils REQUIRED)
add_executable(expr main.cc ${CMAKE_CURRENT_SOURCE_DIR}/../src/STDesc.cpp)
target_include_directories(expr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(expr
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  expr-utils::dataloader
  yaml-cpp::yaml-cpp)
install(TARGETS expr DESTINATION lib/${PROJECT_NAME})
