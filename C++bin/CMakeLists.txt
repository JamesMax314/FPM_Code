cmake_minimum_required(VERSION 3.14)
project(C__bin)

set(CMAKE_CXX_STANDARD 14)

#FIND_PACKAGE(OpenCV REQUIRED)

add_library(C__bin library.cpp library.h)

#target_include_directories(C__bin PUBLIC ${OpenCV_INCLUDE_DIRS})
