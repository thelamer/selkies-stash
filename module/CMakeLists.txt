cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

project(screen_capture_module)

find_package(X11 REQUIRED)
find_package(JPEG REQUIRED)
find_package(Threads REQUIRED)

include_directories(${X11_INCLUDE_DIR} ${JPEG_INCLUDE_DIRS})
link_directories(${X11_LIBRARY_DIR} ${JPEG_LIBRARY_DIRS})

add_library(screen_capture_module SHARED
    screen_capture_module.cpp
    src/xxhash.c
)

target_link_libraries(screen_capture_module
    PUBLIC
    ${X11_LIBRARIES}
    ${JPEG_LIBRARIES}
    Threads::Threads
)

set_target_properties(screen_capture_module PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
