cmake_minimum_required(VERSION 2.6)

set(PROJECT_NAME face_detection)

project(${PROJECT_NAME})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

find_package(OpenCV REQUIRED)
set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

SET(CMAKE_BUILD_TYPE Debug)
ADD_DEFINITIONS(-Wall -Wextra -std=c++11 -Wno-comment -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable)

add_executable(
  ${PROJECT_NAME}
  face_detection.cpp
  eyelash_detection.cpp
  header.hpp
  main.cpp
  tracking.cpp
  lips_detection.cpp
)


TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
