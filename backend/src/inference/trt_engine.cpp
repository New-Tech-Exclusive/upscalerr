#include "trt_engine.h"
#include <fstream>
#include <iostream>

TrtEngine::TrtEngine() {
    m_Runtime = nullptr;
    m_Engine = nullptr;
    m_Context = nullptr;
    m_GraphInitialized = false;
}

TrtEngine::~TrtEngine() {
    Cleanup();
}

bool TrtEngine::Load(const std::string& enginePath) {
    Cleanup();

    // Read engine file
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[TensorRT] Failed to open engine file: " << enginePath << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "[TensorRT] Failed to read engine data." << std::endl;
        return false;
    }

    // 1. Create runtime
    m_Runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_Logger));
    if (!m_Runtime) return false;

    // 2. Deserialize engine
    m_Engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        m_Runtime->deserializeCudaEngine(buffer.data(), size),
        [](nvinfer1::ICudaEngine* engine) { if (engine) delete engine; }
    );
    if (!m_Engine) {
        std::cerr << "[TensorRT] Failed to deserialize CUDA Engine." << std::endl;
        return false;
    }

    // 3. Create context
    m_Context = std::shared_ptr<nvinfer1::IExecutionContext>(
        m_Engine->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ctx) { if (ctx) delete ctx; }
    );
    if (!m_Context) return false;

    std::cout << "[TensorRT] Loaded Engine: " << enginePath << std::endl;
    return true;
}

int TrtEngine::GetBindingIndex(const std::string& tensorName) {
    if (!m_Engine) return -1;
    // TensorRT 8.5/10.0 compatible binding lookup
    for (int i = 0; i < m_Engine->getNbIOTensors(); ++i) {
        const char* name = m_Engine->getIOTensorName(i);
        if (std::string(name) == tensorName) {
            return i;
        }
    }
    return -1;
}

bool TrtEngine::SetInputDimensions(const std::string& inputName, int batch, int channels, int height, int width) {
    if (!m_Context) return false;

    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0] = batch;
    dims.d[1] = channels;
    dims.d[2] = height;
    dims.d[3] = width;

    // In TensorRT 8.5+, we set input dimensions on the execution context
    const char* tensorName = inputName.c_str();
    if (!m_Context->setInputShape(tensorName, dims)) {
        std::cerr << "[TensorRT] Failed to set input dimensions for: " << inputName << std::endl;
        return false;
    }
    return true;
}

bool TrtEngine::AllocateBuffers() {
    if (!m_Engine || !m_Context) return false;

    // Clean up previous buffers
    for (auto& buf : m_Buffers) {
        if (buf.devicePtr) {
            cudaFree(buf.devicePtr);
        }
    }
    m_Buffers.clear();
    m_BindingPtrs.clear();

    int nbTensors = m_Engine->getNbIOTensors();
    m_BindingPtrs.resize(nbTensors, nullptr);

    for (int i = 0; i < nbTensors; ++i) {
        const char* tensorName = m_Engine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_Engine->getTensorIOMode(tensorName);
        nvinfer1::DataType dataType = m_Engine->getTensorDataType(tensorName);

        // Fetch current execution dimensions
        nvinfer1::Dims dims = m_Context->getTensorShape(tensorName);

        size_t numElements = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            numElements *= dims.d[d];
        }

        size_t typeSize = 4; // float32
        if (dataType == nvinfer1::DataType::kHALF) {
            typeSize = 2; // float16
        } else if (dataType == nvinfer1::DataType::kINT8) {
            typeSize = 1;
        }

        TensorBuffer buffer;
        buffer.name = tensorName;
        buffer.isInput = (ioMode == nvinfer1::TensorIOMode::kINPUT);
        buffer.dataType = dataType;
        buffer.sizeInBytes = numElements * typeSize;

        // Allocate Device memory
        cudaError_t err = cudaMalloc(&buffer.devicePtr, buffer.sizeInBytes);
        if (err != cudaSuccess) {
            std::cerr << "[TensorRT] Buffer allocation failed: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        // Enforce zero initialization
        cudaMemset(buffer.devicePtr, 0, buffer.sizeInBytes);

        // Bind in context
        m_Context->setTensorAddress(tensorName, buffer.devicePtr);

        m_Buffers.push_back(buffer);
        m_BindingPtrs[i] = buffer.devicePtr;
    }

    return true;
}

bool TrtEngine::Enqueue(cudaStream_t stream) {
    if (!m_Context) return false;

    // EnqueueV3 handles new TensorRT shape and context updates
    if (!m_Context->enqueueV3(stream)) {
        std::cerr << "[TensorRT] Execution failed." << std::endl;
        return false;
    }
    return true;
}

bool TrtEngine::InitializeGraph(cudaStream_t stream) {
    if (!m_Context) return false;

    if (m_GraphInitialized) {
        if (m_GraphExec) {
            cudaGraphExecDestroy(m_GraphExec);
            m_GraphExec = nullptr;
        }
        if (m_Graph) {
            cudaGraphDestroy(m_Graph);
            m_Graph = nullptr;
        }
        m_GraphInitialized = false;
    }

    // Warm up the engine first
    if (!m_Context->enqueueV3(stream)) return false;
    cudaStreamSynchronize(stream);

    // Capture CUDA Graph
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return false;

    if (!m_Context->enqueueV3(stream)) {
        cudaStreamEndCapture(stream, nullptr);
        return false;
    }

    err = cudaStreamEndCapture(stream, &m_Graph);
    if (err != cudaSuccess || !m_Graph) return false;

    err = cudaGraphInstantiate(&m_GraphExec, m_Graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) return false;

    m_GraphInitialized = true;
    std::cout << "[TensorRT] CUDA Graph captured and instantiated successfully." << std::endl;
    return true;
}

bool TrtEngine::LaunchGraph(cudaStream_t stream) {
    if (!m_GraphInitialized || !m_GraphExec) return false;

    cudaError_t err = cudaGraphLaunch(m_GraphExec, stream);
    return (err == cudaSuccess);
}

void* TrtEngine::GetInputBuffer(const std::string& inputName) {
    for (const auto& buf : m_Buffers) {
        if (buf.isInput && buf.name == inputName) {
            return buf.devicePtr;
        }
    }
    return nullptr;
}

void* TrtEngine::GetOutputBuffer(const std::string& outputName) {
    for (const auto& buf : m_Buffers) {
        if (!buf.isInput && buf.name == outputName) {
            return buf.devicePtr;
        }
    }
    return nullptr;
}

nvinfer1::Dims TrtEngine::GetInputDims(const std::string& inputName) {
    if (!m_Context) return {};
    return m_Context->getTensorShape(inputName.c_str());
}

nvinfer1::Dims TrtEngine::GetOutputDims(const std::string& outputName) {
    if (!m_Context) return {};
    return m_Context->getTensorShape(outputName.c_str());
}

void TrtEngine::Cleanup() {
    if (m_GraphExec) {
        cudaGraphExecDestroy(m_GraphExec);
        m_GraphExec = nullptr;
    }
    if (m_Graph) {
        cudaGraphDestroy(m_Graph);
        m_Graph = nullptr;
    }
    m_GraphInitialized = false;

    for (auto& buf : m_Buffers) {
        if (buf.devicePtr) {
            cudaFree(buf.devicePtr);
        }
    }
    m_Buffers.clear();
    m_BindingPtrs.clear();

    m_Context = nullptr;
    m_Engine = nullptr;
    m_Runtime = nullptr;
}
