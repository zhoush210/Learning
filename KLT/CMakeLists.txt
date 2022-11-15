cmake_minimum_required(VERSION 2.8)
project(KLT)

find_package(OpenCV REQUIRED)

add_executable(lkdemo lkdemo.cpp)
target_link_libraries(lkdemo ${OpenCV_LIBS})

add_executable(klt klt.cpp)
target_link_libraries(klt ${OpenCV_LIBS})