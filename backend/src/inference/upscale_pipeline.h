#pragma once

#include <string>
#include <memory>
#include <cuda_runtime.h>
#include "trt_engine.h"

class UpscalePipeline {
public:
    UpscalePipeline();
    ~UpscalePipeline();

    bool Initialize(int scaleFactor, int inputWidth, int inputHeight);
    void Cleanup();

    // Run the main upscaling pipeline: converts rawSurface -> TRT input -> inf -> Vulkan present surface
    bool Process(cudaSurfaceObject_t rawSurface, cudaSurfaceObject_t presentSurface);

    // Direct copy for disabled upscaling
    bool CopyDirect(cudaSurfaceObject_t rawSurface, cudaSurfaceObject_t presentSurface);

private:
    std::unique_ptr<TrtEngine> m_TrtEngine;
    cudaStream_t m_Stream = nullptr;

    int m_ScaleFactor = 2;
    int m_InputWidth = 0;
    int m_InputHeight = 0;
    int m_OutputWidth = 0;
    int m_OutputHeight = 0;

    // Buffer references to device memory owned/registered by TrtEngine
    void* m_InputDevBuffer = nullptr;
    void* m_OutputDevBuffer = nullptr;
};
