# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(map_trailGT_project)

#set(OpenCV_DIR /media/12_04/home/cer/Documents/ud/arv_s2016/software/opencv-3.1.0)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#SET(GENERIC_INCLUDE_DIRS  "/usr/include")
#SET(OPENCV2_LIB_DIRS  "/usr/lib/opencv2")

#message(STATUS "    cv2 libraries: ${OpenCV2_LIBS}")
#message(STATUS "    generic include path: ${GENERIC_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(map_trailGT trailGT.cpp tinyxml.cpp tinystr.cpp tinyxmlerror.cpp tinyxmlparser.cpp)

# Link your application with OpenCV libraries
target_link_libraries(map_trailGT ${OpenCV2_LIBS} ${OpenCV_LIBS})
