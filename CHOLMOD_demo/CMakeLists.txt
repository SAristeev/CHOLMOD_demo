cmake_minimum_required (VERSION 3.8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)

find_package(CUDAToolkit REQUIRED)
find_package(SuiteSparse CONFIG REQUIRED)

add_executable (${PROJECT_NAME} "${PROJECT_NAME}.c" "CHOLMOD_demo.h")

target_link_libraries(${PROJECT_NAME} PRIVATE ${SuiteSparse_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart )