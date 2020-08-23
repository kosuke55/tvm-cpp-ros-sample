cmake_minimum_required(VERSION 3.0.2)
project(tl_tvm)

add_compile_options(-std=c++14)

set(TVM_ROOT /home/kosuke/tvm)
set(DMLC_CORE ${TVM_ROOT}/3rdparty/dmlc-core)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -fPIC")
# not needed ?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${TVM_ROOT}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${DMLC_CORE}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${TVM_ROOT}/3rdparty/dlpack/include")
set (CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-ltvm_runtime -L${TVM_ROOT}/build -ldl -pthread")


find_package(catkin REQUIRED COMPONENTS
  nodelet
  pluginlib
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES tl_tvm
 CATKIN_DEPENDS nodelet pluginlib roscpp sensor_msgs
 DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/tl_tvm_nodelet.cpp
)

target_link_libraries(${PROJECT_NAME}
  tvm_runtime
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES})
