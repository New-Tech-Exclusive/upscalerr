#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log warnings and errors by default
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    bool Load(const std::string& enginePath);
    void Cleanup();

    // Sets dynamic shape binding dimensions on the profile
    bool SetInputDimensions(const std::string& inputName, int batch, int channels, int height, int width);
    
    // Allocates CUDA device memory for bindings based on resolved engine dimensions
    bool AllocateBuffers();

    // Perform TensorRT model inference
    bool Enqueue(cudaStream_t stream);

    // CUDA Graph interface for sub-2ms overhead
    bool InitializeGraph(cudaStream_t stream);
    bool LaunchGraph(cudaStream_t stream);

    void* GetInputBuffer(const std::string& inputName);
    void* GetOutputBuffer(const std::string& outputName);

    nvinfer1::Dims GetInputDims(const std::string& inputName);
    nvinfer1::Dims GetOutputDims(const std::string& outputName);

private:
    Logger m_Logger;
    std::shared_ptr<nvinfer1::IRuntime> m_Runtime = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine> m_Engine = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> m_Context = nullptr;

    // Buffer structure
    struct TensorBuffer {
        std::string name;
        void* devicePtr = nullptr;
        size_t sizeInBytes = 0;
        bool isInput = false;
        nvinfer1::DataType dataType;
    };

    std::vector<TensorBuffer> m_Buffers;
    std::vector<void*> m_BindingPtrs; // Pointers index-matched for Context enqueue

    // CUDA Graph state variables
    cudaGraph_t m_Graph = nullptr;
    cudaGraphExec_t m_GraphExec = nullptr;
    bool m_GraphInitialized = false;

    int GetBindingIndex(const std::string& tensorName);
};
