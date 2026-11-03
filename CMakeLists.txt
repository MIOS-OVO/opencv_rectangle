cmake_minimum_required(VERSION 3.16)

project(rectangle_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(rect_detect main.cpp)

target_include_directories(rect_detect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rect_detect PRIVATE ${OpenCV_LIBS})
