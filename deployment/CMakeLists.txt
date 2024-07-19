cmake_minimum_required(VERSION 3.4.1)

project(test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# set to release 
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

#CUDA
set(CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)
find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda/lib64)
list(APPEND ALL_INCLUDE ${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "Find CUDA headers at ${CUDA_INCLUDE_DIRS}")


# tensorrt api
set(TENSORRT_PATH /workingspace/tensorrt/TensorRT-8.6.1.6)
find_library(TENSORRT_LIB  NAMES libnvinfer.so PATHS ${TENSORRT_PATH}/lib)
include_directories(${TENSORRT_PATH}/include)

# opencv

set(OpenCV_ROOT /usr)
include_directories(${OpenCV_ROOT}/include/opencv4)
find_library(CORE NAMES libopencv_core.so PATHS ${OpenCV_ROOT}/lib/x86_64-linux-gnu)
find_library(HIGHGUI NAMES libopencv_highgui.so PATHS ${OpenCV_ROOT}/lib/x86_64-linux-gnu)
find_library(IMGPROC NAMES libopencv_imgproc.so PATHS ${OpenCV_ROOT}/lib/x86_64-linux-gnu)
find_library(CODEC NAMES libopencv_imgcodecs.so PATHS ${OpenCV_ROOT}/lib/x86_64-linux-gnu)
find_library(JPEG NAMES libjpeg.so PATHS ${OpenCV_ROOT}/lib/x86_64-linux-gnu)


set(OPENCVLIBS ${CORE} ${HIGHGUI} ${IMGPROC} ${CODEC} ${JPEG})


# rtdetr so file
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB LIB_SOURCES src/*.cpp)
add_library(Rtdetr SHARED ${LIB_SOURCES})
target_link_libraries(Rtdetr ${TENSORRT_LIB} ${OPENCVLIBS})

# EXE
include_directories(${CMAKE_SOURCE_DIR}/test)
include_directories(${CMAKE_SOURCE_DIR}/test/ini)
include_directories(${CMAKE_SOURCE_DIR}/test/otl)
file(GLOB SOURCES 
    test/ini/*.c
    test/*.cpp
    test/otl/thread/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE Rtdetr ${OPENCVLIBS} cudart pthread)
