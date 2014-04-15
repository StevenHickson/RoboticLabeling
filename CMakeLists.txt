cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(RoboticLabeling)

SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

IF(WIN32 OR win64)
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll)
ELSE()
   SET(CMAKE_FIND_LIBRARY_SUFFIXES .a)
ENDIF()


find_package(PCL 1.7 REQUIRED)
find_package( OpenCV REQUIRED )

add_subdirectory(src)

set(KinectSDK_INCLUDE "default value" CACHE FILEPATH "Include directories for Kinect2 SDK")
set(KinectSDK_LIB "default value" CACHE FILEPATH "Include directories for Kinect2 SDK")

include_directories(C:/opencv/build)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${RoboticLabeling_SOURCE_DIR}/include)
include_directories(../PCL_Kinect2SDK/include)
include_directories(${KinectSDK_INCLUDE})

link_directories(${PCL_LIBRARY_DIRS})
link_directories(../../PCL_Kinect2SDK/build/)
link_directories(${KinectSDK_LIB})

add_definitions(${PCL_DEFINITIONS})

file(GLOB_RECURSE RoboticLabeling_HEADERS include/*.h)
file(GLOB_RECURSE RoboticLabeling_SOURCES src/*.cpp)
set (RoboticLabeling_INCLUDE_DIRS "")
foreach (_headerFile ${RoboticLabeling_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND RoboticLabeling_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES RoboticLabeling_INCLUDE_DIRS)

include_directories(${RoboticLabeling_INCLUDE_DIRS})

add_executable (RoboticLabeling ${RoboticLabeling_SOURCES} ${RoboticLabeling_HEADERS})
target_link_libraries (RoboticLabeling ${PCL_LIBRARIES} ${OpenCV_LIBS} Kinect20.lib debug Debug/PCL_Kinect2SDK optimized Release/PCL_Kinect2SDK)