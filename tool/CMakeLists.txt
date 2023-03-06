set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV 3 REQUIRED)
include_directories($OpenCV_INCLUDE_DIRS)

add_executable(draw draw.cpp)
target_link_libraries(draw ${OpenCV_LIBS})