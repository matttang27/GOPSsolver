cmake_minimum_required(VERSION 3.16)
project(CppSolver VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

# GLPK via vcpkg (recommended on Windows)
find_path(GLPK_INCLUDE_DIR glpk.h)
find_library(GLPK_LIBRARY NAMES glpk)

if(NOT GLPK_INCLUDE_DIR OR NOT GLPK_LIBRARY)
    message(FATAL_ERROR "GLPK not found. Ensure vcpkg glpk is installed and the vcpkg toolchain is used.")
endif()

# Add your executable
file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_executable(cpp_solver ${SOURCES})

target_include_directories(cpp_solver PRIVATE "${GLPK_INCLUDE_DIR}")
target_link_libraries(cpp_solver PRIVATE "${GLPK_LIBRARY}")
