cmake_minimum_required(VERSION 3.18)
project(boltzmann_smoke LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

# CUDA part (object library)
add_library(boltzmann_cuda_obj OBJECT src/boltzmann_solver.cu)
target_include_directories(boltzmann_cuda_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(boltzmann_cuda_obj PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# C++ part (exclude boltzmann_solver.cpp)
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
list(REMOVE_ITEM CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/boltzmann_solver.cu")

add_executable(boltzmann_smoke ${CPP_SOURCES} $<TARGET_OBJECTS:boltzmann_cuda_obj>)

# Required libraries
find_package(OpenVDB CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find OpenGL related libraries
find_package(OpenGL REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# ImGui
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC
    OpenGL::GL
    glfw
)

# Include directories (OpenVDB etc. are often automatically added during linking, so can be omitted)
target_include_directories(boltzmann_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(boltzmann_smoke
    PRIVATE
    OpenVDB::openvdb
    TBB::tbb
    Imath::Imath
    CUDA::cudart
    OpenGL::GL
    GLEW::GLEW
    glfw
    imgui
)
