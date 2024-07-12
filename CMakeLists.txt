cmake_minimum_required(VERSION 3.18)
project(model_deploy)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(TVM_ROOT "/opt/tvm")
set(DMLC_CORE "${TVM_ROOT}/3rdparty/dmlc-core")
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(TVM_RUNTIME_LIB "${TVM_ROOT}/build/libtvm_runtime.so")

include_directories(${TVM_ROOT}/include)
include_directories(${DMLC_CORE}/include)
include_directories(${TVM_ROOT}/3rdparty/dlpack/include)

# cuda 11.1 is recommended .
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(CUDA libs: ${CUDA_LIBRARIES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(model_deploy deploy_yolov5n.cc)
target_link_libraries(model_deploy 
    ${CUDA_LIBRARIES}
    ${TVM_RUNTIME_LIB}
    ${OpenCV_LIBRARIES}
    rt 
    cuda)
