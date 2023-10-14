cmake_minimum_required(VERSION 3.26)
project(Efficient_DNN)

set(CMAKE_CXX_STANDARD 17)

add_executable(Efficient_DNN main.cpp)
add_executable(b img2col_conv.cpp)