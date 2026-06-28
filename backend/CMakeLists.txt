cmake_minimum_required(VERSION 3.20)
project(upscalerr_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Release build flags if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA Architectures configuration
# Target Ampere (sm_80, sm_86), Ada Lovelace (sm_89), and Blackwell (sm_120)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 120)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile options for MSVC (only apply to CXX, not CUDA/NVCC)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/WX>
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/MP>
)

# Locate packages
find_package(CUDA REQUIRED)
find_package(Vulkan REQUIRED)

# Find TensorRT
# Fixed environment variable token lookup and updated library names for TensorRT 11.x
if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR "$ENV{TENSORRT_DIR}")
else()
    set(TENSORRT_DIR "C:/TensorRT" "C:/Program Files/NVIDIA/TensorRT")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS ${TENSORRT_DIR}
    PATH_SUFFIXES include
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer_11 nvinfer
    PATHS ${TENSORRT_DIR}
    PATH_SUFFIXES lib x64/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_DIR environment variable pointing to the TensorRT installation directory.")
endif()

# Windows API libraries required for DX12, WGC, and IPC
set(WIN32_LIBS
    d3d11.lib
    d3d12.lib
    dxgi.lib
    dwmapi.lib
    User32.lib
    Gdi32.lib
    WindowsApp.lib
    CoreMessaging.lib
)

# Backend target
add_executable(upscalerr_backend
    src/main.cpp
    src/capture/wgc_capture.h
    src/capture/wgc_capture.cpp
    src/capture/dx12_cuda_interop.h
    src/capture/dx12_cuda_interop.cu
    src/inference/trt_engine.h
    src/inference/trt_engine.cpp
    src/inference/upscale_pipeline.h
    src/inference/upscale_pipeline.cu
    src/inference/framegen_pipeline.h
    src/inference/framegen_pipeline.cu
    src/kernels/color_convert.cu
    src/kernels/pixel_shuffle.cu
    src/kernels/bilinear_warp.cu
    src/kernels/blend_output.cu
    src/presentation/vulkan_swapchain.h
    src/presentation/vulkan_swapchain.cpp
    src/presentation/vk_cuda_interop.h
    src/presentation/vk_cuda_interop.cu
    src/ipc/protocol.h
    src/ipc/pipe_server.h
    src/ipc/pipe_server.cpp
)

# Include directories
target_include_directories(upscalerr_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(upscalerr_backend PRIVATE
    ${CUDA_LIBRARIES}
    cuda
    cudart
    ${Vulkan_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${WIN32_LIBS}
)

# Set runtime output directories
set_target_properties(upscalerr_backend PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
)

# Copy TensorRT DLLs next to executable for runtime loading (Windows specific)
if(WIN32)
    find_file(TENSORRT_DLL 
        NAMES nvinfer_11.dll nvinfer_10.dll nvinfer.dll
        PATHS ${TENSORRT_DIR}
        PATH_SUFFIXES bin lib
    )
    if(TENSORRT_DLL)
        add_custom_command(TARGET upscalerr_backend POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TENSORRT_DLL}"
            $<TARGET_FILE_DIR:upscalerr_backend>
        )
    endif()
endif()
