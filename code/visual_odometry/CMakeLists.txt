cmake_minimum_required(VERSION 3.10)
project(CornerDetection)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Print OpenCV version for debugging
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Add executable
add_executable(corner_detection main_visualization.cpp)

# Include OpenCV headers
target_include_directories(corner_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV libraries
target_link_libraries(corner_detection PRIVATE ${OpenCV_LIBS})

# Optional: Set compiler warnings
if(MSVC)
    target_compile_options(corner_detection PRIVATE /W4)
else()
    target_compile_options(corner_detection PRIVATE -Wall -Wextra)
endif()

# Optional: Set build type if not defined
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optional: Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")