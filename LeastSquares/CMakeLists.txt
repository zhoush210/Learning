cmake_minimum_required(VERSION 2.8)
project(LeastSquares)

find_package(OpenCV REQUIRED)

add_executable(gauss_newton gauss_newton.cpp)
target_link_libraries(gauss_newton ${OpenCV_LIBS})